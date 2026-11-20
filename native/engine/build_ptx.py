"""Compile the engine kernels to PTX with NVRTC (no host compiler needed)."""
import argparse, ctypes as C, os, sys
from pathlib import Path

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_PATH_V12_6", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")))
    ap.add_argument("--source", type=Path, default=Path(__file__).with_name("kernels.cu"))
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("build") / "kernels.ptx")
    args = ap.parse_args()
    cuda = args.cuda.resolve()
    with os.add_dll_directory(str(cuda / "bin")):
        lib = C.WinDLL(str(cuda / "bin" / "nvrtc64_120_0.dll"))
    def fn(name, res, args_):
        f = getattr(lib, name); f.restype = res; f.argtypes = args_; return f
    errstr = fn("nvrtcGetErrorString", C.c_char_p, [C.c_int])
    def check(r):
        if r != 0: raise RuntimeError(errstr(r).decode())
    prog_t = C.c_void_p
    create = fn("nvrtcCreateProgram", C.c_int, [C.POINTER(prog_t), C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
    compile_ = fn("nvrtcCompileProgram", C.c_int, [prog_t, C.c_int, C.POINTER(C.c_char_p)])
    logsize = fn("nvrtcGetProgramLogSize", C.c_int, [prog_t, C.POINTER(C.c_size_t)])
    getlog = fn("nvrtcGetProgramLog", C.c_int, [prog_t, C.c_char_p])
    ptxsize = fn("nvrtcGetPTXSize", C.c_int, [prog_t, C.POINTER(C.c_size_t)])
    getptx = fn("nvrtcGetPTX", C.c_int, [prog_t, C.c_char_p])
    destroy = fn("nvrtcDestroyProgram", C.c_int, [C.POINTER(prog_t)])
    work = Path("C:/idolbuild")
    work.mkdir(parents=True, exist_ok=True)
    local = work / "kernels.cu"
    local.write_bytes(args.source.read_bytes())
    prog = prog_t()
    check(create(C.byref(prog), local.read_bytes(), b"kernels.cu", 0, None, None))
    flags = [
        b"--gpu-architecture=compute_86",
        b"--std=c++17",
        b"-default-device",
        b"--use_fast_math",
        b"-I" + str(cuda / "include").encode(),
    ]
    arr = (C.c_char_p * len(flags))(*flags)
    status = compile_(prog, len(flags), arr)
    n = C.c_size_t(); check(logsize(prog, C.byref(n)))
    log = C.create_string_buffer(n.value + 1); check(getlog(prog, log))
    text = log.value.decode("utf-8", "replace")
    if text.strip(): print(text)
    check(status)
    check(ptxsize(prog, C.byref(n)))
    buf = C.create_string_buffer(n.value); check(getptx(prog, buf))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(buf.raw[:n.value])
    destroy(C.byref(prog))
    print("wrote", args.output, n.value, "bytes")

if __name__ == "__main__":
    main()


