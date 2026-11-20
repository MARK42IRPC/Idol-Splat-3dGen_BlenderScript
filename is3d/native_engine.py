"""ctypes front end for the bundled IS3D inference engine.

The engine is a self-contained binary: it talks to ``nvcuda.dll`` from the NVIDIA driver and
streams the bundled weight file, so the add-on needs no torch, no CUDA toolkit and no ONNX
runtime.  Everything here is plain Python so it can also be driven from a normal interpreter
(the Blender specific image loading lives in ``worker.py``).
"""

import ctypes as C
import time
from pathlib import Path

ABI = 4
SOFT_LIMIT_MIB = 4096.0
HARD_LIMIT_MIB = 6144.0
MIN_FREE_MIB = 2600.0

_F32 = C.c_float
_P32 = C.POINTER(C.c_float)
_U64 = C.c_uint64
_PROGRESS = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)

STAGE_MESSAGES = {
    "preprocess": "正在裁切图像并去除背景",
    "encode": "正在提取图像特征",
    "denoise": "正在采样高斯模型",
    "octree": "正在解码高斯锚点",
    "gaussians": "正在生成高斯点",
    "export": "正在写入高斯 PLY",
    "done": "高斯模型已生成",
}


class NativeEngineError(RuntimeError):
    """Raised when the native engine cannot run on this machine."""


def package_assets():
    """Path to the engine binary and weights shipped inside the add-on."""
    base = Path(__file__).resolve().with_name("native")
    dll, weights = base / "idol_engine.dll", base / "idol_weights.bin"
    if dll.is_file() and weights.is_file():
        return dll, weights
    # Running straight from the source checkout uses the build output instead of a packaged copy.
    build = Path(__file__).resolve().parents[1] / "native" / "engine" / "build"
    return build / "idol_engine.dll", build / "idol_weights.bin"


def check_assets(dll=None, weights=None):
    dll, weights = (Path(dll), Path(weights)) if dll and weights else package_assets()
    if not dll.is_file():
        raise NativeEngineError(f"缺少推理引擎 {dll}")
    if not weights.is_file():
        raise NativeEngineError(f"缺少模型权重 {weights}")
    return dll, weights


class Engine:
    """One loaded engine instance; call :meth:`close` (or use it as a context manager)."""

    def __init__(self, dll=None, weights=None, device=0):
        self.dll_path, self.weights_path = check_assets(dll, weights)
        self._lib = C.WinDLL(str(self.dll_path))
        self._bind()
        abi = self._lib.idol_engine_abi()
        if abi != ABI:
            raise NativeEngineError(f"推理引擎 ABI {abi} 与插件要求的 {ABI} 不一致")
        self._handle = C.c_void_p()
        rc = self._lib.idol_engine_create(device, str(self.weights_path).encode("utf-8"), C.byref(self._handle))
        if rc or not self._handle:
            raise NativeEngineError(self._error())
        self._callback = None

    # -- plumbing ---------------------------------------------------------
    def _bind(self):
        lib = self._lib
        lib.idol_engine_abi.restype = C.c_uint32
        lib.idol_engine_create.argtypes = [C.c_int32, C.c_char_p, C.POINTER(C.c_void_p)]
        lib.idol_engine_create.restype = C.c_int32
        lib.idol_engine_destroy.argtypes = [C.c_void_p]
        lib.idol_engine_info.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int32),
                                         C.POINTER(C.c_int32), C.POINTER(_U64), C.POINTER(_U64)]
        lib.idol_engine_info.restype = C.c_int32
        lib.idol_memory.argtypes = [C.c_void_p, C.POINTER(_U64), C.POINTER(_U64)]
        lib.idol_memory.restype = C.c_int32
        lib.idol_engine_reset_peak.argtypes = [C.c_void_p]
        lib.idol_engine_unload.argtypes = [C.c_void_p]
        lib.idol_engine_set_crop_margin.argtypes = [C.c_void_p, C.c_float]
        lib.idol_engine_set_crop_margin.restype = C.c_int32
        lib.idol_engine_set_remove_background.argtypes = [C.c_void_p, C.c_int32]
        lib.idol_engine_set_remove_background.restype = C.c_int32
        lib.idol_generate.argtypes = [C.c_void_p, _P32, C.c_int, C.c_int, _U64, C.c_int, C.c_int,
                                      C.c_char_p, C.c_char_p, _PROGRESS, C.c_void_p]
        lib.idol_generate.restype = C.c_int32
        lib.idol_engine_last_error.restype = C.c_char_p

    def _error(self):
        text = self._lib.idol_engine_last_error()
        return text.decode("utf-8", "replace") if text else "推理引擎报告了未知错误"

    # -- queries ----------------------------------------------------------
    def info(self):
        if not self._handle:
            raise NativeEngineError("推理引擎已关闭")
        name = C.create_string_buffer(256)
        major, minor = C.c_int32(), C.c_int32()
        free_b, total_b = _U64(), _U64()
        rc = self._lib.idol_engine_info(self._handle, name, 256, C.byref(major), C.byref(minor),
                                        C.byref(free_b), C.byref(total_b))
        if rc:
            raise NativeEngineError(self._error())
        return {
            "device": name.value.decode("utf-8", "replace"),
            "compute_capability": f"{major.value}.{minor.value}",
            "free_mib": free_b.value / 2 ** 20,
            "total_mib": total_b.value / 2 ** 20,
        }

    def memory(self):
        """Bytes this engine currently holds on the device, and its peak so far (MiB)."""
        if not self._handle:
            return 0.0, 0.0
        used, peak = _U64(), _U64()
        self._lib.idol_memory(self._handle, C.byref(used), C.byref(peak))
        return used.value / 2 ** 20, peak.value / 2 ** 20

    def reset_peak(self):
        if self._handle:
            self._lib.idol_engine_reset_peak(self._handle)

    def set_crop_margin(self, margin):
        if self._handle and self._lib.idol_engine_set_crop_margin(self._handle, float(margin)):
            raise NativeEngineError(self._error())

    def set_remove_background(self, enabled=True):
        """Cut the subject out with the bundled BiRefNet matting pass before encoding."""
        if self._handle and self._lib.idol_engine_set_remove_background(self._handle, 1 if enabled else 0):
            raise NativeEngineError(self._error())

    # -- work -------------------------------------------------------------
    def generate(self, rgba, width, height, ply_path, camera_path=None, seed=42, steps=50,
                 points=262144, progress=None):
        """Run the whole image-to-Gaussians pipeline and write a Gaussian PLY."""
        if not self._handle:
            raise NativeEngineError("推理引擎已关闭")
        ply_path = Path(ply_path)
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        speaker = None
        if progress is not None:
            def _bridge(percent, message, _user):
                name = message.decode("utf-8", "replace") if message else ""
                if name == "sampling":
                    done = max(1, min(int(steps), round((int(percent) - 12) * steps / 43)))
                    progress(int(percent), f"正在采样高斯模型（{done}/{steps} 步）")
                else:
                    progress(int(percent), STAGE_MESSAGES.get(name, name))

            self._callback = _PROGRESS(_bridge)
            speaker = self._callback
        started = time.perf_counter()
        rc = self._lib.idol_generate(
            self._handle, rgba.ctypes.data_as(_P32), int(width), int(height), int(seed), int(steps),
            int(points), str(ply_path).encode("utf-8"),
            str(camera_path).encode("utf-8") if camera_path else None, speaker, None)
        seconds = time.perf_counter() - started
        if rc:
            raise NativeEngineError(self._error())
        used, peak = self.memory()
        return {"seconds": seconds, "peak_mib": peak, "resident_mib": used}

    def unload(self):
        """Drop every weight and activation buffer held by the engine."""
        if self._handle:
            self._lib.idol_engine_unload(self._handle)
            self._lib.idol_engine_destroy(self._handle)
            self._handle = C.c_void_p()

    close = unload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.unload()
        return False


def budget_report(peak_mib):
    """Classify a measured peak against the add-on's VRAM budget."""
    if peak_mib > HARD_LIMIT_MIB:
        return {"status": "error",
                "message": f"显存峰值 {peak_mib:.0f} MiB 超过 {HARD_LIMIT_MIB:.0f} MiB 上限"}
    if peak_mib > SOFT_LIMIT_MIB:
        return {"status": "warning",
                "message": f"显存峰值 {peak_mib:.0f} MiB 超过 {SOFT_LIMIT_MIB:.0f} MiB 预算"}
    return {"status": "ok", "message": f"显存峰值 {peak_mib:.0f} MiB"}
