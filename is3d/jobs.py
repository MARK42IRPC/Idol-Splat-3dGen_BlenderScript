"""Run the Blender worker without touching the editor's Blender API."""

from collections import deque
import json
import locale
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
from uuid import uuid4


_PROGRESS_PREFIX = b"IS3D_PROGRESS "


def _attach_windows_job(process):
    """Return a close callback that terminates this worker and its descendants."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time_limit", ctypes.c_longlong), ("job_time_limit", ctypes.c_longlong),
            ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic_limits", BasicLimits), ("io_counters", ctypes.c_ulonglong * 6),
            ("process_memory_limit", ctypes.c_size_t), ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = ExtendedLimits()
    limits.basic_limits.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    process_handle = int(process._handle)  # The exact Win32 handle owned by CPython's Popen.
    if (kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            and kernel.AssignProcessToJobObject(handle, process_handle)):
        return lambda: kernel.CloseHandle(handle)
    kernel.CloseHandle(handle)
    return None


def _end_worker(process, close_job):
    if close_job is not None:
        close_job()
    elif os.name != "nt":
        # The worker starts in a new session. Its PID names that session's process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        # Some hosts prevent assigning nested Job Objects. Scope the fallback to this PID.
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.kill()
    return process.wait(timeout=5)


def _result_stamp(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return (info.st_size, info.st_mtime_ns, info.st_ino) if path.is_file() and info.st_size else None


class Job:
    """One worker invocation. Poll queue events on the Blender main thread."""

    def __init__(self, binary_path, worker_path, job_json, output_dir):
        self.binary_path = Path(binary_path).resolve()
        self.worker_path = Path(worker_path).resolve()
        self.job_json = Path(job_json).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.result_path = self.output_dir / "asset_transfer.blend"
        self.log_path = self.output_dir / "worker.log"
        self._events = queue.SimpleQueue()
        self._cancel = threading.Event()
        self._thread = None
        self._tail = deque(maxlen=12)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("This IS3D job has already been started")
        self._thread = threading.Thread(target=self._run, name="IS3D worker supervisor", daemon=True)
        self._thread.start()
        return self

    def poll_events(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def cancel(self):
        self._cancel.set()

    def close(self, timeout=0.0):
        self.cancel()
        if timeout > 0 and self._thread is not None:
            self._thread.join(timeout)
        return not self.running

    def _receive(self, raw):
        if raw.startswith(_PROGRESS_PREFIX):
            try:
                payload = json.loads(raw[len(_PROGRESS_PREFIX):])
                percent = float(payload["percent"])
                message = payload["message"]
                if math.isfinite(percent) and 0 <= percent <= 100 and isinstance(message, str):
                    self._events.put({"kind": "progress", "percent": percent, "message": message})
                    return
            except (ValueError, KeyError, TypeError):
                pass
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode(locale.getpreferredencoding(False), errors="replace")
        if text and text.strip() != "Blender quit":
            self._tail.append(text[-500:])

    def _read_log(self, reader, pending, *, final=False):
        while chunk := reader.read(65536):
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                self._receive(line.rstrip(b"\r"))
            if len(pending) > 65536:
                self._receive(pending)
                pending = b""
        if final and pending:
            self._receive(pending.rstrip(b"\r"))
        return pending

    def _run(self):
        terminal = {"kind": "cancelled", "message": "IS3D generation cancelled", "returncode": None}
        try:
            if self._cancel.is_set():
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                self.log_path = self.output_dir / f"worker-{uuid4().hex[:8]}.log"
            previous_result = _result_stamp(self.result_path)
            command = [
                str(self.binary_path), "--background", "--factory-startup", "--python-exit-code", "1",
                "--python", str(self.worker_path), "--", "--job", str(self.job_json),
            ]
            platform = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            with self.log_path.open("xb") as output, self.log_path.open("rb") as reader:
                process = subprocess.Popen(
                    command, cwd=self.job_json.parent, env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, **platform,
                )
                close_job = None
                try:
                    if os.name == "nt":
                        close_job = _attach_windows_job(process)
                    pending = b""
                    while not self._cancel.is_set() and process.poll() is None:
                        pending = self._read_log(reader, pending)
                        self._cancel.wait(0.075)
                finally:
                    terminal["returncode"] = _end_worker(process, close_job)
                self._read_log(reader, pending, final=True)
            if self._cancel.is_set():
                return
            if terminal["returncode"] != 0:
                detail = "\n".join(self._tail)
                raise RuntimeError(f"Blender worker exited with code {terminal['returncode']}.\n{detail}".strip())
            result = _result_stamp(self.result_path)
            if result is None or result == previous_result:
                raise RuntimeError("Blender worker exited without writing a new asset_transfer.blend")
            terminal.update(kind="finished", message="IS3D asset generated", result_path=str(self.result_path))
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            terminal.update(kind="error", message=str(error))
        finally:
            terminal["log_path"] = str(self.log_path)
            self._events.put(terminal)
