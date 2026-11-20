#include "idol_engine.h"

#define NOMINMAX
#include <windows.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ptx_embed.inc"

namespace {

constexpr uint32_t kAbi = 4;
// Dynamic shared memory for the 64-row flash tile: Q 8K + K 8K + V 8K + S 16K + P 8K + softmax rows.
// A/B staging (24 KiB) and the 128x64 fp32 accumulator tile (32 KiB) share one window.
constexpr size_t kGemmSmem = 128 * 64 * 4;
constexpr size_t kFlashSmem = 64 * 64 * 2 + 2 * 64 * 64 * 2 + 64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4;
thread_local char g_error[2048] = {};
bool g_debug_sync = false;
bool g_debug_pool = false;
bool g_profile = false;
double g_qpc_freq = 1.0;

// Per-kernel timings, only filled when IDOL_PROFILE is set (each launch is bracketed by a sync).
struct ProfRow {
    double secs = 0.0;
    unsigned long long calls = 0;
};
std::unordered_map<std::string, ProfRow> g_prof;
double g_last_launch_secs = 0.0;

double now_sec() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / g_qpc_freq;
}

void prof_dump(const char* tag) {
    if (g_prof.empty()) return;
    std::vector<std::pair<std::string, ProfRow>> rows(g_prof.begin(), g_prof.end());
    std::sort(rows.begin(), rows.end(),
              [](const std::pair<std::string, ProfRow>& a, const std::pair<std::string, ProfRow>& b) {
                  return a.second.secs > b.second.secs;
              });
    double total = 0.0;
    for (const auto& r : rows) total += r.second.secs;
    fprintf(stderr, "IDOL_PROFILE %s kernels=%.3fs\n", tag, total);
    for (const auto& r : rows) {
        fprintf(stderr, "  %-22s %8.3fs %8llu calls %6.1f%%\n", r.first.c_str(), r.second.secs,
                r.second.calls, total > 0.0 ? 100.0 * r.second.secs / total : 0.0);
    }
    fflush(stderr);
    g_prof.clear();
}
// Exact accounting of the device memory this engine holds, independent of whatever else runs on the GPU.
uint64_t g_dev_bytes = 0;
uint64_t g_dev_peak = 0;

void dev_add(size_t bytes) {
    g_dev_bytes += bytes;
    if (g_dev_bytes > g_dev_peak) g_dev_peak = g_dev_bytes;
}

void dev_sub(size_t bytes) { g_dev_bytes = g_dev_bytes > bytes ? g_dev_bytes - bytes : 0; }

int32_t fail(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, args);
    va_end(args);
    return 1;
}

const char* wf(const char* fmt, ...) {
    static thread_local char bufs[8][256];
    static thread_local int idx = 0;
    char* b = bufs[idx];
    idx = (idx + 1) & 7;
    va_list args;
    va_start(args, fmt);
    vsnprintf(b, 256, fmt, args);
    va_end(args);
    return b;
}

#define DRIVER_ENTRIES(X) \
    X(init, cuInit, "cuInit") \
    X(device_get, cuDeviceGet, "cuDeviceGet") \
    X(device_name, cuDeviceGetName, "cuDeviceGetName") \
    X(device_attr, cuDeviceGetAttribute, "cuDeviceGetAttribute") \
    X(ctx_create, cuCtxCreate, "cuCtxCreate_v2") \
    X(ctx_destroy, cuCtxDestroy, "cuCtxDestroy_v2") \
    X(ctx_push, cuCtxPushCurrent, "cuCtxPushCurrent_v2") \
    X(ctx_pop, cuCtxPopCurrent, "cuCtxPopCurrent_v2") \
    X(sync, cuCtxSynchronize, "cuCtxSynchronize") \
    X(module_load, cuModuleLoadDataEx, "cuModuleLoadDataEx") \
    X(module_unload, cuModuleUnload, "cuModuleUnload") \
    X(module_fn, cuModuleGetFunction, "cuModuleGetFunction") \
    X(fn_attr, cuFuncSetAttribute, "cuFuncSetAttribute") \
    X(mem_alloc, cuMemAlloc, "cuMemAlloc_v2") \
    X(mem_free, cuMemFree, "cuMemFree_v2") \
    X(htod, cuMemcpyHtoD, "cuMemcpyHtoD_v2") \
    X(dtoh, cuMemcpyDtoH, "cuMemcpyDtoH_v2") \
    X(dtod, cuMemcpyDtoD, "cuMemcpyDtoD_v2") \
    X(memset, cuMemsetD8, "cuMemsetD8_v2") \
    X(meminfo, cuMemGetInfo, "cuMemGetInfo_v2") \
    X(launch, cuLaunchKernel, "cuLaunchKernel") \
    X(err_name, cuGetErrorName, "cuGetErrorName") \
    X(err_string, cuGetErrorString, "cuGetErrorString")

struct Driver {
    HMODULE lib = nullptr;
#define M(member, fn, name) decltype(&fn) member = nullptr;
    DRIVER_ENTRIES(M)
#undef M
    bool load() {
        lib = LoadLibraryA("nvcuda.dll");
        if (!lib) return false;
#define B(member, fn, name) member = reinterpret_cast<decltype(member)>(GetProcAddress(lib, name)); if (!member) return false;
        DRIVER_ENTRIES(B)
#undef B
        return true;
    }
    ~Driver() { if (lib) FreeLibrary(lib); }
};

struct Weights {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE map = nullptr;
    const uint8_t* base = nullptr;
    size_t bytes = 0;
    struct Entry { const uint8_t* data; uint64_t nbytes; int rank; int dims[4]; int dtype; };
    std::unordered_map<std::string, Entry> index;

    bool open(const char* path) {
        int wn = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
        std::wstring wide(wn > 0 ? wn : 0, L'\0');
        if (wn <= 0 || !MultiByteToWideChar(CP_UTF8, 0, path, -1, wide.data(), wn)) return false;
        file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER size;
        if (!GetFileSizeEx(file, &size)) return false;
        bytes = (size_t)size.QuadPart;
        map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map) return false;
        base = (const uint8_t*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
        if (!base) return false;
        if (bytes < 24 || memcmp(base, "IDOLW1\0\0", 8) != 0) return false;
        uint32_t count;
        uint64_t data_off;
        memcpy(&count, base + 8, 4);
        memcpy(&data_off, base + 16, 8);
        const uint8_t* p = base + 24;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t nlen;
            memcpy(&nlen, p, 4); p += 4;
            std::string name((const char*)p, nlen); p += nlen;
            int rank; uint32_t dims[4] = {0, 0, 0, 0}; int dtype; uint64_t off, nelem;
            memcpy(&rank, p, 4); p += 4;
            for (int d = 0; d < rank; ++d) { memcpy(&dims[d], p, 4); p += 4; }
            memcpy(&dtype, p, 4); p += 4;
            memcpy(&off, p, 8); p += 8;
            memcpy(&nelem, p, 8); p += 8;
            Entry e{};
            e.data = base + data_off + off;
            e.nbytes = nelem * (dtype == 0 ? 2 : 4);
            e.rank = rank;
            memcpy(e.dims, dims, sizeof(e.dims));
            e.dtype = dtype;
            index.emplace(std::move(name), e);
        }
        return true;
    }
    const Entry* find(const char* name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    ~Weights() {
        if (base) UnmapViewOfFile(base);
        if (map) CloseHandle(map);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

}  // namespace

struct Tensor {
    CUdeviceptr p = 0;
    int r = 0;
    int c = 0;
    size_t bytes = 0;
    bool ok() const { return p != 0; }
};

// Simple size-class pool: released blocks are reused, so the activation footprint stays bounded.
struct Pool {
    struct Block { CUdeviceptr p; size_t bytes; };
    std::vector<Block> live;
    std::vector<Block> cache;
    size_t used = 0, peak = 0;

    CUdeviceptr alloc(size_t bytes) {
        bytes = (bytes + 1023) / 1024 * 1024;
        for (size_t i = 0; i < cache.size(); ++i) {
            if (cache[i].bytes == bytes) {
                CUdeviceptr p = cache[i].p;
                cache[i] = cache.back();
                cache.pop_back();
                used += bytes;
                if (used > peak) peak = used;
                if (g_debug_pool) fprintf(stderr, "A %llx %zu\n", (unsigned long long)p, bytes);
                return p;
            }
        }
        CUdeviceptr p = 0;
        if (cuMemAlloc(&p, bytes) != CUDA_SUCCESS) return 0;
        live.push_back({p, bytes});
        dev_add(bytes);
        used += bytes;
        if (used > peak) peak = used;
        if (g_debug_pool) fprintf(stderr, "A %llx %zu\n", (unsigned long long)p, bytes);
        return p;
    }
    void release(CUdeviceptr p, size_t bytes) {
        bytes = (bytes + 1023) / 1024 * 1024;
        used -= bytes;
        if (g_debug_pool) fprintf(stderr, "R %llx %zu\n", (unsigned long long)p, bytes);
        for (auto& b : cache)
            if (b.p == p) fprintf(stderr, "[pool] DOUBLE RELEASE %llx (%zu bytes)\n",
                                  (unsigned long long)p, bytes);
        cache.push_back({p, bytes});
    }
    void clear() {
        for (auto& b : live) {
            cuMemFree(b.p);
            dev_sub(b.bytes);
        }
        // `cache` only mirrors blocks that are still listed in `live`, so freeing `live` covers them.
        live.clear();
        cache.clear();
        used = 0;
    }
    // Drop cached (unused) blocks so a finished stage hands its memory back to the driver.
    void trim() {
        for (auto& b : cache) {
            cuMemFree(b.p);
            dev_sub(b.bytes);
            live.erase(std::remove_if(live.begin(), live.end(),
                                      [&](const Block& x) { return x.p == b.p; }),
                       live.end());
        }
        cache.clear();
    }
    ~Pool() { clear(); }
};

struct IdolEngine {
    Driver drv;
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    Weights weights;
    Pool pool;
    struct WBlock {
        CUdeviceptr p;
        size_t bytes;
        std::string name;
    };
    std::vector<WBlock> wblocks;
    std::unordered_map<std::string, CUdeviceptr> wmap;
    std::unordered_map<std::string, CUfunction> kernels;
    char dev_name[256] = {};
    int cc_major = 0, cc_minor = 0;
    uint64_t peak_bytes = 0;
    uint64_t base_bytes = 0;
    float crop_margin = 1.2f;
    int32_t remove_background = 1;

    CUfunction kernel(const char* name, size_t smem = 0) {
        auto it = kernels.find(name);
        if (it != kernels.end()) return it->second;
        CUfunction fn = nullptr;
        if (cuModuleGetFunction(&fn, mod, name) != CUDA_SUCCESS) return nullptr;
        if (smem > 48 * 1024) cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem);
        kernels.emplace(name, fn);
        return fn;
    }

    int32_t launch(const char* name, dim3 grid, dim3 block, size_t smem, void** args) {
        CUfunction fn = kernel(name, smem);
        if (!fn) return fail("missing kernel %s", name);
        double t0 = 0.0;
        if (g_profile) {
            cuCtxSynchronize();
            t0 = now_sec();
        }
        CUresult r = cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, nullptr, args, nullptr);
        if (g_profile) {
            cuCtxSynchronize();
            const double dt = now_sec() - t0;
            ProfRow& row = g_prof[name];
            row.secs += dt;
            row.calls += 1;
            g_last_launch_secs = dt;
        } else {
            g_last_launch_secs = 0.0;
        }
        if (r != CUDA_SUCCESS) return fail("launch %s failed (%d)", name, (int)r);
        if (g_debug_sync) {
            CUresult r2 = cuCtxSynchronize();
            if (r2 != CUDA_SUCCESS) {
                const char* s = nullptr;
                cuGetErrorString(r2, &s);
                return fail("kernel %s faulted: %s", name, s ? s : "?");
            }
        }
        return 0;
    }

    int32_t sync() {
        CUresult r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) {
            const char* s = nullptr;
            cuGetErrorString(r, &s);
            return fail("cuda sync failed: %s", s ? s : "?");
        }
        uint64_t free_b = 0, total_b = 0;
        cuMemGetInfo(&free_b, &total_b);
        uint64_t used = total_b - free_b;
        used = used > base_bytes ? used - base_bytes : 0;
        if (used > peak_bytes) peak_bytes = used;
        return 0;
    }

    const Weights::Entry* we(const char* name) const { return weights.find(name); }

    CUdeviceptr wget(const char* name) {
        auto it = wmap.find(name);
        if (it != wmap.end()) return it->second;
        const Weights::Entry* e = weights.find(name);
        if (!e) return 0;
        size_t bytes = (size_t)((e->nbytes + 1023) / 1024 * 1024);
        CUdeviceptr p = 0;
        CUresult r = cuMemAlloc(&p, bytes);
        if (r != CUDA_SUCCESS) {
            const char* s = nullptr;
            cuGetErrorString(r, &s);
            fail("cuMemAlloc %s (%llu bytes): %s", name, (unsigned long long)bytes, s ? s : "?");
            return 0;
        }
        wblocks.push_back({p, bytes});
        wblocks.back().name = name;
        dev_add(bytes);
        r = cuMemcpyHtoD(p, e->data, e->nbytes);
        if (r != CUDA_SUCCESS) {
            const char* s = nullptr;
            cuGetErrorString(r, &s);
            fail("cuMemcpyHtoD %s: %s", name, s ? s : "?");
            return 0;
        }
        wmap.emplace(name, p);
        return p;
    }

    void free_weights() {
        if (!wblocks.empty()) cuCtxSynchronize();
        for (auto& b : wblocks) {
            cuMemFree(b.p);
            dev_sub(b.bytes);
        }
        wblocks.clear();
        wmap.clear();
    }

    // Release every weight whose name starts with `prefix` (e.g. one transformer block) so the
    // streaming stages only ever keep a single block resident on the device.
    void free_weights_prefix(const char* prefix) {
        if (wblocks.empty()) return;
        cuCtxSynchronize();
        const std::string head(prefix);
        for (size_t i = 0; i < wblocks.size();) {
            if (wblocks[i].name.compare(0, head.size(), head) == 0) {
                cuMemFree(wblocks[i].p);
                dev_sub(wblocks[i].bytes);
                wmap.erase(wblocks[i].name);
                wblocks[i] = wblocks.back();
                wblocks.pop_back();
            } else {
                ++i;
            }
        }
    }

    Tensor tm(int rows, int cols) {
        Tensor t;
        int rp = rows > 0 ? (rows + 127) / 128 * 128 : 128;
        int cp = cols > 0 ? (cols + 63) / 64 * 64 : 64;
        t.bytes = (size_t)rp * cp * 2;
        t.p = pool.alloc(t.bytes);
        t.r = rp;
        t.c = cp;
        return t;
    }
    Tensor tmx(int rows, int cols) {
        Tensor t;
        t.bytes = (size_t)rows * cols * 2;
        t.p = pool.alloc(t.bytes);
        t.r = rows;
        t.c = cols;
        return t;
    }
    void rel(Tensor& t) {
        if (t.p) { pool.release(t.p, t.bytes); t.p = 0; }
    }
    void zero(Tensor& t, int64_t n) {
        if (n <= 0) return;
        int nn = (int)n;
        void* a[] = {&t.p, &nn};
        launch("k_zero", dim3((nn + 255) / 256), dim3(256), 0, a);
    }
    void zero_tail(Tensor& t, int valid_rows) {
        if (t.r <= valid_rows) return;
        CUdeviceptr p = t.p + (size_t)valid_rows * t.c * 2;
        int n = (t.r - valid_rows) * t.c;
        void* a[] = {&p, &n};
        launch("k_zero", dim3((n + 255) / 256), dim3(256), 0, a);
    }

    int gemm_raw(CUdeviceptr a, int lda, int M, int K, CUdeviceptr wp, int N, CUdeviceptr bp,
                 CUdeviceptr dst, int ldc, int act) {
        int tiles_n = (N + 63) / 64;
        dim3 grid(((M + 127) / 128) * tiles_n);
        void* args[] = {&a, &lda, &wp, &N, &bp, &dst, &ldc, &M, &N, &K, &act};
        int rc = launch("k_gemm2", grid, dim3(256), kGemmSmem, args);
        if (g_profile) {
            char key[64];
            snprintf(key, sizeof key, "  gemm %dx%dx%d", M, K, N);
            ProfRow& row = g_prof[key];
            row.secs += g_last_launch_secs;
            row.calls += 1;
        }
        return rc;
    }

    int gemm_to(Tensor a, const char* wname, const char* bname, int act, CUdeviceptr dst, int ldc) {
        const Weights::Entry* e = weights.find(wname);
        if (!e) return fail("missing weight %s", wname);
        int K = e->dims[0], N = e->dims[1];
        if (a.c < K) return fail("gemm %s: lda %d < K %d", wname, a.c, K);
        CUdeviceptr wp = wget(wname);
        if (!wp) return 1;
        CUdeviceptr bp = 0;
        if (bname) { bp = wget(bname); if (!bp) return 1; }
        return gemm_raw(a.p, a.c, a.r, K, wp, N, bp, dst, ldc, act);
    }

    Tensor gemm(Tensor a, const char* wname, const char* bname, int act) {
        const Weights::Entry* e = weights.find(wname);
        if (!e) { fail("missing weight %s", wname); return Tensor{}; }
        Tensor out = tm(a.r, e->dims[1]);
        if (!out.ok()) { fail("oom in gemm %s", wname); return Tensor{}; }
        if (gemm_to(a, wname, bname, act, out.p, out.c)) { rel(out); return Tensor{}; }
        return out;
    }

    // LayerNorm (fp32 math) with optional affine weights and optional adaLN scale/shift, in place.
    int ln_to(Tensor x, int rows, int cols, const char* wname, const char* bname, float eps,
              CUdeviceptr ms, CUdeviceptr mt, CUdeviceptr dst) {
        CUdeviceptr wp = 0, bp = 0;
        if (wname) { wp = wget(wname); if (!wp) return fail("missing %s", wname); }
        if (bname) { bp = wget(bname); if (!bp) return fail("missing %s", bname); }
        CUdeviceptr src = x.p, dp = dst;
        int ldx = x.c, ldy = x.c;
        int z = 0;
        void* args[] = {&src, &ldx, &wp, &bp, &dp, &ldy, &rows, &cols, &eps, &ms, &mt};
        return launch("k_ln", dim3(rows), dim3(256), 0, args);
    }
    int ln(Tensor x, int rows, int cols, const char* wname, const char* bname, float eps) {
        return ln_to(x, rows, cols, wname, bname, eps, 0, 0, x.p);
    }

    int rms(Tensor t, int rows, int heads, const char* gamma) {
        CUdeviceptr gp = wget(gamma);
        if (!gp) return fail("missing %s", gamma);
        int dim = 64;
        float scale = 8.0f;
        void* args[] = {&t.p, &gp, &rows, &heads, &dim, &scale};
        return launch("k_rms_heads", dim3(rows, (heads + 7) / 8), dim3(256), 0, args);
    }

    Tensor flash(Tensor q, Tensor k, Tensor v, int Lq, int Lk, int H) {
        Tensor o = tm(Lq, H * 64);
        if (!o.ok()) { fail("oom in flash"); return Tensor{}; }
        float scale = 1.0f / std::sqrt(64.0f);
        void* args[] = {&q.p, &k.p, &v.p, &o.p, &Lq, &Lk, &H, &q.c, &o.c, &scale};
        if (launch("k_flash3", dim3((Lq + 63) / 64, H), dim3(128), kFlashSmem, args)) { rel(o); return Tensor{}; }
        if (g_profile) {
            char key[64];
            snprintf(key, sizeof key, "  flash L%dxH%d", Lq, H);
            ProfRow& row = g_prof[key];
            row.secs += g_last_launch_secs;
            row.calls += 1;
        }
        return o;
    }

    ~IdolEngine() {
        free_weights();
        pool.clear();
        if (mod && ctx) {
            cuCtxPushCurrent(ctx);
            cuModuleUnload(mod);
            cuCtxPopCurrent(nullptr);
            mod = nullptr;
        }
        if (ctx) { cuCtxDestroy(ctx); ctx = nullptr; }
    }
};

namespace {

void add_t(IdolEngine& E, Tensor& a, Tensor b) {
    int n = a.r * a.c;
    void* p[] = {&a.p, &b.p, &n};
    E.launch("k_add", dim3((n + 255) / 256), dim3(256), 0, p);
}

Tensor copy_t(IdolEngine& E, Tensor x) {
    Tensor y = E.tm(x.r, x.c);
    if (!y.ok()) { fail("oom copy"); return Tensor{}; }
    int n = x.r * x.c;
    void* p[] = {&y.p, &x.p, &n};
    E.launch("k_copy_h2h", dim3((n + 255) / 256), dim3(256), 0, p);
    return y;
}

void act_t(IdolEngine& E, Tensor x, int kind, int64_t n) {
    int nn = (int)n;
    void* p[] = {&x.p, &nn, &kind};
    E.launch("k_act", dim3((nn + 255) / 256), dim3(256), 0, p);
}

Tensor image_nhwc(IdolEngine& E, const float* prepared, float s0, float s1, float s2,
                  float b0, float b1, float b2) {
    Tensor img = E.tmx(1024 * 1024, 3);
    if (!img.ok()) { fail("oom image"); return Tensor{}; }
    int H = 1024, W = 1024;
    float a0 = s0, a1 = s1, a2 = s2, c0 = b0, c1 = b1, c2 = b2;
    size_t plane_bytes = (size_t)3 * 1024 * 1024 * 4;
    CUdeviceptr ph = E.pool.alloc(plane_bytes);
    if (!ph) { fail("oom image upload"); E.rel(img); return Tensor{}; }
    if (E.drv.htod(ph, prepared, plane_bytes) != CUDA_SUCCESS) {
        fail("image upload failed");
        E.pool.release(ph, plane_bytes);
        E.rel(img);
        return Tensor{};
    }
    void* p[] = {&ph, &img.p, &H, &W, &a0, &a1, &a2, &c0, &c1, &c2};
    int rc = E.launch("k_image_nhwc", dim3((H * W * 3 + 255) / 256), dim3(256), 0, p);
    E.pool.release(ph, plane_bytes);
    if (rc) { E.rel(img); return Tensor{}; }
    return img;
}

Tensor conv2d(IdolEngine& E, Tensor x, int H, int W, int C, const char* wname, const char* bname,
              int KH, int KW, int stride, int pad, int act, int Hout = 0, int Wout = 0) {
    const Weights::Entry* e = E.weights.find(wname);
    if (!e) { fail("missing %s", wname); return Tensor{}; }
    int Kp = e->dims[0], N = e->dims[1];
    int Ktrue = C * KH * KW;
    if (!Hout) Hout = (H + 2 * pad - KH) / stride + 1;
    if (!Wout) Wout = (W + 2 * pad - KW) / stride + 1;
    Tensor out = E.tm(Hout * Wout, N);
    if (!out.ok()) { fail("oom conv %s", wname); return Tensor{}; }
    CUdeviceptr wp = E.wget(wname);
    CUdeviceptr bp = bname ? E.wget(bname) : 0;
    if (!wp || (bname && !bp)) { fail("upload %s", wname); E.rel(out); return Tensor{}; }
    const int CH = 2048;
    const int total = Hout * Wout;
    for (int p0 = 0; p0 < total; p0 += CH) {
        int n = (total - p0 < CH) ? (total - p0) : CH;
        Tensor A = E.tm(n, Ktrue);
        if (!A.ok()) { fail("oom im2col"); E.rel(out); return Tensor{}; }
        if (Kp != Ktrue) E.zero(A, (int64_t)A.r * A.c);
        void* a1[] = {&x.p, &A.p, &A.c, &H, &W, &C, &KH, &KW, &stride, &pad, &Hout, &Wout, &p0, &n};
        if (E.launch("k_im2col_s", dim3((n * Ktrue + 255) / 256), dim3(256), 0, a1) ||
            E.gemm_raw(A.p, A.c, n, Kp, wp, N, bp, out.p + (size_t)p0 * N * 2, N, act)) {
            E.rel(A);
            E.rel(out);
            return Tensor{};
        }
        E.rel(A);
    }
    return out;
}

int gn(IdolEngine& E, Tensor& x, int HW, int C, const char* gw, const char* gb, int groups,
       float eps, int silu_after) {
    CUdeviceptr g = gw ? E.wget(gw) : 0;
    CUdeviceptr b = gb ? E.wget(gb) : 0;
    if (gw && !g) return fail("missing %s", gw);
    CUdeviceptr mean = E.pool.alloc(groups * 4);
    CUdeviceptr rstd = E.pool.alloc(groups * 4);
    if (!mean || !rstd) return fail("oom gn");
    int H = HW, one = 1;
    void* a1[] = {&x.p, &mean, &rstd, &H, &one, &C, &groups, &eps};
    if (E.launch("k_groupnorm_stats", dim3(groups), dim3(256), 256 * 4, a1)) return 1;
    void* a2[] = {&x.p, &mean, &rstd, &g, &b, &x.p, &HW, &C, &groups};
    if (E.launch("k_groupnorm_apply", dim3((HW * C + 255) / 256), dim3(256), 0, a2)) return 1;
    E.pool.release(mean, groups * 4);
    E.pool.release(rstd, groups * 4);
    if (silu_after) act_t(E, x, 1, (int64_t)HW * C);
    return 0;
}

Tensor resnet(IdolEngine& E, Tensor x, int H, int W, int Cin, int Cout, const char* pre) {
    char wn[384], bn[384];
    Tensor skip;
    if (Cin != Cout) {
        snprintf(wn, sizeof wn, "%s.conv_shortcut.weight", pre);
        snprintf(bn, sizeof bn, "%s.conv_shortcut.bias", pre);
        skip = conv2d(E, x, H, W, Cin, wn, bn, 1, 1, 1, 0, 0);
    } else {
        skip = copy_t(E, x);
    }
    if (!skip.ok()) return Tensor{};
    snprintf(wn, sizeof wn, "%s.norm1.weight", pre);
    snprintf(bn, sizeof bn, "%s.norm1.bias", pre);
    if (gn(E, x, H * W, Cin, wn, bn, 32, 1e-6f, 1)) { E.rel(skip); return Tensor{}; }
    snprintf(wn, sizeof wn, "%s.conv1.weight", pre);
    snprintf(bn, sizeof bn, "%s.conv1.bias", pre);
    Tensor c1 = conv2d(E, x, H, W, Cin, wn, bn, 3, 3, 1, 1, 0);
    E.rel(x);
    if (!c1.ok()) { E.rel(skip); return Tensor{}; }
    snprintf(wn, sizeof wn, "%s.norm2.weight", pre);
    snprintf(bn, sizeof bn, "%s.norm2.bias", pre);
    if (gn(E, c1, H * W, Cout, wn, bn, 32, 1e-6f, 1)) { E.rel(skip); E.rel(c1); return Tensor{}; }
    snprintf(wn, sizeof wn, "%s.conv2.weight", pre);
    snprintf(bn, sizeof bn, "%s.conv2.bias", pre);
    Tensor c2 = conv2d(E, c1, H, W, Cout, wn, bn, 3, 3, 1, 1, 0);
    E.rel(c1);
    if (!c2.ok()) { E.rel(skip); return Tensor{}; }
    add_t(E, c2, skip);
    E.rel(skip);
    return c2;
}

int vae_attention(IdolEngine& E, Tensor& x, int HW, int C, const char* pre) {
    char wn[384], bn[384];
    Tensor res = copy_t(E, x);
    if (!res.ok()) return 1;
    snprintf(wn, sizeof wn, "%s.group_norm.weight", pre);
    snprintf(bn, sizeof bn, "%s.group_norm.bias", pre);
    if (gn(E, x, HW, C, wn, bn, 32, 1e-5f, 0)) { E.rel(res); return 1; }
    snprintf(wn, sizeof wn, "%s.to_q.weight", pre);
    snprintf(bn, sizeof bn, "%s.to_q.bias", pre);
    Tensor q = E.gemm(x, wn, bn, 0);
    snprintf(wn, sizeof wn, "%s.to_k.weight", pre);
    snprintf(bn, sizeof bn, "%s.to_k.bias", pre);
    Tensor k = E.gemm(x, wn, bn, 0);
    snprintf(wn, sizeof wn, "%s.to_v.weight", pre);
    snprintf(bn, sizeof bn, "%s.to_v.bias", pre);
    Tensor v = E.gemm(x, wn, bn, 0);
    E.rel(x);
    if (!q.ok() || !k.ok() || !v.ok()) { E.rel(q); E.rel(k); E.rel(v); E.rel(res); return fail("vae attn gemm"); }
    float sc = 1.0f / std::sqrt((float)C);
    int nq = q.r * q.c;
    void* s1[] = {&q.p, &nq, &sc};
    E.launch("k_scale", dim3((nq + 255) / 256), dim3(256), 0, s1);

    const int CHQ = 512;
    Tensor attn = E.tm(HW, C);
    CUdeviceptr S = E.pool.alloc((size_t)CHQ * HW * 4);
    Tensor P = E.tm(CHQ, HW);
    if (!S || !attn.ok() || !P.ok()) return fail("oom vae attn workspace");
    for (int p0 = 0; p0 < HW; p0 += CHQ) {
        int n = (HW - p0 < CHQ) ? (HW - p0) : CHQ;
        CUdeviceptr qp = q.p + (size_t)p0 * q.c * 2;
        void* g1[] = {&qp, &q.c, &k.p, &k.c, &S, &HW, &n, &HW, &C};
        if (E.launch("k_gemm_nt", dim3(HW / 32, (n + 31) / 32), dim3(256), 0, g1)) return 1;
        void* g2[] = {&S, &n, &HW};
        if (E.launch("k_softmax_rows", dim3(n), dim3(256), 0, g2)) return 1;
        int np = n * HW;
        void* g3[] = {&S, &HW, &P.p, &P.c, &n, &HW};
        if (E.launch("k_f32_to_f16_rows", dim3((np + 255) / 256), dim3(256), 0, g3)) return 1;
        if (E.gemm_raw(P.p, P.c, n, HW, v.p, C, 0, attn.p + (size_t)p0 * attn.c * 2, attn.c, 0)) return 1;
    }
    E.pool.release((CUdeviceptr)S, (size_t)CHQ * HW * 4);
    E.rel(P);
    E.rel(q);
    E.rel(k);
    E.rel(v);
    snprintf(wn, sizeof wn, "%s.to_out.0.weight", pre);
    snprintf(bn, sizeof bn, "%s.to_out.0.bias", pre);
    Tensor o = E.gemm(attn, wn, bn, 0);
    E.rel(attn);
    if (!o.ok()) { E.rel(res); return 1; }
    add_t(E, o, res);
    E.rel(res);
    x = o;
    return 0;
}

static std::wstring g_dino_dump_dir;
static int g_dino_dbg_layer = 0;

static std::vector<float> dino_fetch(IdolEngine& E, Tensor t, const char* tag) {
    int rows = t.r, cols = t.c;
    size_t bytes = (size_t)rows * (size_t)cols * 4;
    std::vector<float> h((size_t)rows * (size_t)cols);
    CUdeviceptr out = E.pool.alloc(bytes);
    if (!out) return h;
    void* a[] = {&t.p, &t.c, &out, &cols, &rows, &cols};
    E.launch("k_f16_to_f32_rows", dim3((unsigned)((bytes / 4 + 255) / 256)), dim3(256), 0, a);
    E.drv.dtoh(h.data(), out, bytes);
    E.pool.release(out, bytes);
    double mx = 0.0;
    size_t nan = 0, inf = 0;
    for (float f : h) {
        if (std::isnan(f)) ++nan;
        else if (std::isinf(f)) ++inf;
        else mx = std::max(mx, (double)std::fabs(f));
    }
    fprintf(stderr, "[dino] %-12s max %.2f nan %zu inf %zu\n", tag, mx, nan, inf);
    return h;
}

static void dino_save_raw(const char* name, const std::vector<float>& h) {
    if (g_dino_dump_dir.empty()) return;
    std::wstring wide(name, name + strlen(name));
    wide = g_dino_dump_dir + L"/" + wide + L".bin";
    FILE* f = _wfopen(wide.c_str(), L"wb");
    if (!f) { fprintf(stderr, "[dino] cannot write dump %s\n", name); return; }
    fprintf(stderr, "[dino] wrote %s (%zu floats)\n", name, h.size());
    fwrite(h.data(), sizeof(float), h.size(), f);
    fclose(f);
}

Tensor dino_encode(IdolEngine& E, const float* prepared) {
    static const bool dbg = GetEnvironmentVariableA("IDOL_DEBUG_DINO", nullptr, 0) > 0;
    if (dbg && g_dino_dump_dir.empty()) {
        wchar_t buf[MAX_PATH];
        DWORD n = GetEnvironmentVariableW(L"IDOL_DEBUG_DINO_DIR", buf, MAX_PATH);
        if (n > 0 && n < MAX_PATH) g_dino_dump_dir = buf;
        g_dino_dbg_layer = atoi(getenv("IDOL_DEBUG_DINO_LAYER") ? getenv("IDOL_DEBUG_DINO_LAYER") : "0");
    }
    Tensor img = image_nhwc(E, prepared, 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f,
                            -0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f);
    if (!img.ok()) return Tensor{};
    Tensor A = E.tm(4096, 768);
    if (!A.ok()) return Tensor{};
    int H = 1024, W = 1024, C = 3, KH = 16, KW = 16, st = 16, pd = 0, Ho = 64, Wo = 64;
    void* p[] = {&img.p, &A.p, &H, &W, &C, &KH, &KW, &st, &pd, &Ho, &Wo};
    if (E.launch("k_im2col", dim3((4096 * 768 + 255) / 256), dim3(256), 0, p)) return Tensor{};
    E.rel(img);
    Tensor patch = E.gemm(A, "dinov3.embeddings.patch_embeddings.weight",
                          "dinov3.embeddings.patch_embeddings.bias", 0);
    E.rel(A);
    if (!patch.ok()) return Tensor{};

    const int L = 4101, CC = 1280, HH = 20, PRE = 5;
    Tensor tok = E.tm(L, CC);
    if (!tok.ok()) return Tensor{};
    CUdeviceptr cls = E.wget("dinov3.embeddings.cls_token");
    CUdeviceptr reg = E.wget("dinov3.embeddings.register_tokens");
    if (!cls || !reg) { fail("dino tokens"); return Tensor{}; }
    int rows = 1, cols = CC;
    void* c1[] = {&tok.p, &tok.c, &cls, &cols, &rows, &cols};
    E.launch("k_copy_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, c1);
    rows = 4;
    CUdeviceptr dst = tok.p + (size_t)CC * 2;
    void* c2[] = {&dst, &tok.c, &reg, &cols, &rows, &cols};
    E.launch("k_copy_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, c2);
    rows = patch.r;
    dst = tok.p + (size_t)PRE * CC * 2;
    void* c3[] = {&dst, &tok.c, &patch.p, &patch.c, &rows, &cols};
    E.launch("k_copy_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, c3);
    E.rel(patch);

    if (dbg) { auto h = dino_fetch(E, tok, "tokens"); dino_save_raw("p_tokens", h); }

    CUdeviceptr rc = E.wget("dinov3.rope_cos");
    CUdeviceptr rs = E.wget("dinov3.rope_sin");
    if (!rc || !rs) { fail("dino rope"); return Tensor{}; }
    const int nrope = (L - PRE) * HH * 32;
    for (int i = 0; i < 32; ++i) {
        char pre[64];
        snprintf(pre, sizeof pre, "dinov3.layer.%d", i);
        char b[2][384];
        Tensor n1 = E.tm(L, CC);
        if (!n1.ok()) return Tensor{};
        snprintf(b[0], sizeof b[0], "%s.norm1.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.norm1.bias", pre);
        if (E.ln_to(tok, L, CC, b[0], b[1], 1e-5f, 0, 0, n1.p)) return Tensor{};
        if (dbg && i == g_dino_dbg_layer) { auto h = dino_fetch(E, n1, "norm1"); dino_save_raw("l_norm1", h); }
        snprintf(b[0], sizeof b[0], "%s.attention.q_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.attention.q_proj.bias", pre);
        Tensor q = E.gemm(n1, b[0], b[1], 0);
        snprintf(b[0], sizeof b[0], "%s.attention.k_proj.weight", pre);
        Tensor k = E.gemm(n1, b[0], nullptr, 0);
        snprintf(b[0], sizeof b[0], "%s.attention.v_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.attention.v_proj.bias", pre);
        Tensor v = E.gemm(n1, b[0], b[1], 0);
        E.rel(n1);
        if (!q.ok() || !k.ok() || !v.ok()) return Tensor{};
        if (dbg && i == g_dino_dbg_layer) {
            auto hq = dino_fetch(E, q, "q"); dino_save_raw("l_q", hq);
            auto hk = dino_fetch(E, k, "k"); dino_save_raw("l_k", hk);
            auto hv = dino_fetch(E, v, "v"); dino_save_raw("l_v", hv);
        }
        int Lq = L, hh = HH, pre5 = PRE;
        void* ra[] = {&q.p, &rc, &rs, &Lq, &hh, &pre5};
        void* rb[] = {&k.p, &rc, &rs, &Lq, &hh, &pre5};
        E.launch("k_rope_dino", dim3((nrope + 255) / 256), dim3(256), 0, ra);
        E.launch("k_rope_dino", dim3((nrope + 255) / 256), dim3(256), 0, rb);
        Tensor o = E.flash(q, k, v, L, L, HH);
        if (dbg && i == g_dino_dbg_layer) { auto h = dino_fetch(E, o, "attn"); dino_save_raw("l_attn", h); }
        E.rel(q);
        E.rel(k);
        E.rel(v);
        if (!o.ok()) return Tensor{};
        snprintf(b[0], sizeof b[0], "%s.attention.o_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.attention.o_proj.bias", pre);
        Tensor o2 = E.gemm(o, b[0], b[1], 0);
        E.rel(o);
        if (!o2.ok()) return Tensor{};
        if (dbg) { auto h = dino_fetch(E, o2, "attn_proj"); if (i == g_dino_dbg_layer) dino_save_raw("l_attnproj", h); }
        snprintf(b[0], sizeof b[0], "%s.layer_scale1.lambda1", pre);
        CUdeviceptr lam = E.wget(b[0]);
        int cc_cols = CC;
        void* sa[] = {&tok.p, &o2.p, &lam, &tok.r, &cc_cols};
        E.launch("k_add_scaled", dim3((tok.r * CC + 255) / 256), dim3(256), 0, sa);
        E.rel(o2);
        if (dbg) { auto h = dino_fetch(E, tok, "res1"); if (i == g_dino_dbg_layer) dino_save_raw("l_res1", h); }

        Tensor n2 = E.tm(L, CC);
        if (!n2.ok()) return Tensor{};
        snprintf(b[0], sizeof b[0], "%s.norm2.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.norm2.bias", pre);
        if (E.ln_to(tok, L, CC, b[0], b[1], 1e-5f, 0, 0, n2.p)) return Tensor{};
        if (dbg && i == g_dino_dbg_layer) { auto h = dino_fetch(E, n2, "norm2"); dino_save_raw("l_norm2", h); }
        snprintf(b[0], sizeof b[0], "%s.mlp.gate_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.mlp.gate_proj.bias", pre);
        Tensor g = E.gemm(n2, b[0], b[1], 0);
        snprintf(b[0], sizeof b[0], "%s.mlp.up_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.mlp.up_proj.bias", pre);
        Tensor u = E.gemm(n2, b[0], b[1], 0);
        E.rel(n2);
        if (!g.ok() || !u.ok()) return Tensor{};
        if (dbg) {
            auto hg = dino_fetch(E, g, "gate"); auto hu = dino_fetch(E, u, "up");
            if (i == g_dino_dbg_layer) { dino_save_raw("l_gate", hg); dino_save_raw("l_up", hu); }
        }
        int gs = g.r * g.c;
        void* ma[] = {&g.p, &u.p, &g.p, &gs};
        E.launch("k_mul_silu", dim3((gs + 255) / 256), dim3(256), 0, ma);
        if (dbg) { auto h = dino_fetch(E, g, "act"); if (i == g_dino_dbg_layer) dino_save_raw("l_act", h); }
        E.rel(u);
        snprintf(b[0], sizeof b[0], "%s.mlp.down_proj.weight", pre);
        snprintf(b[1], sizeof b[1], "%s.mlp.down_proj.bias", pre);
        Tensor d = E.gemm(g, b[0], b[1], 0);
        E.rel(g);
        if (!d.ok()) return Tensor{};
        snprintf(b[0], sizeof b[0], "%s.layer_scale2.lambda1", pre);
        lam = E.wget(b[0]);
        void* sa2[] = {&tok.p, &d.p, &lam, &tok.r, &cc_cols};
        E.launch("k_add_scaled", dim3((tok.r * CC + 255) / 256), dim3(256), 0, sa2);
        E.rel(d);
        if (dbg) {
            auto h = dino_fetch(E, tok, "res2");
            if (i < 8) {
                char nm[32];
                snprintf(nm, sizeof nm, "h%d", i + 1);
                dino_save_raw(nm, h);
            }
            if (i == g_dino_dbg_layer) dino_save_raw("l_res2", h);
        }
        // Stream the trunk: only the current block stays resident, so the encoder needs ~50 MiB of
        // weights on the device instead of the full 1.6 GiB.
        E.free_weights_prefix(wf("%s.", pre));
    }
    E.ln(tok, L, CC, "dinov3.norm.weight", "dinov3.norm.bias", 1e-5f);
    E.ln(tok, L, CC, nullptr, nullptr, 1e-5f);
    return tok;
}

static std::wstring g_vae_dir;
int download_f32(IdolEngine& E, Tensor t, int rows, int cols, float* dst);

static void vae_dump(IdolEngine& E, Tensor x, int H, int W, int C, const char* tag) {
    if (g_vae_dir.empty()) return;
    size_t n = (size_t)H * W * C;
    std::vector<float> h(n, 0.f);
    if (download_f32(E, x, H * W, C, h.data())) { fprintf(stderr, "[vae] %s download failed\n", tag); return; }
    float mn = 1e30f, mx = -1e30f;
    double sum = 0.0;
    size_t bad = 0;
    for (float f : h) {
        if (!std::isfinite(f)) { ++bad; continue; }
        mn = std::min(mn, f); mx = std::max(mx, f); sum += f;
    }
    fprintf(stderr, "[vae] %-14s min %.3f max %.3f mean %.4f bad %zu\n", tag, mn, mx, sum / (double)n, bad);
    std::wstring wname(tag, tag + strlen(tag));
    std::wstring path = g_vae_dir + L"/" + wname + L".bin";
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (f) { fwrite(h.data(), sizeof(float), h.size(), f); fclose(f); }
}

Tensor vae_encode(IdolEngine& E, const float* prepared) {
    if (g_vae_dir.empty()) {
        wchar_t buf[MAX_PATH];
        DWORD nb = GetEnvironmentVariableW(L"IDOL_DEBUG_VAE_DIR", buf, MAX_PATH);
        if (nb > 0 && nb < MAX_PATH) g_vae_dir = buf;
    }
    Tensor img = image_nhwc(E, prepared, 2.f, 2.f, 2.f, -1.f, -1.f, -1.f);
    if (!img.ok()) return Tensor{};
    int H = 1024, W = 1024;
    Tensor x = conv2d(E, img, H, W, 3, "vae.encoder.conv_in.weight", "vae.encoder.conv_in.bias",
                      3, 3, 1, 1, 0);
    E.rel(img);
    if (!x.ok()) return Tensor{};
    vae_dump(E, x, H, W, 128, "conv_in");
    const int ch[4] = {128, 256, 512, 512};
    int cin = 128;
    for (int b = 0; b < 4; ++b) {
        for (int r = 0; r < 2; ++r) {
            char pre[128];
            snprintf(pre, sizeof pre, "vae.encoder.down_blocks.%d.resnets.%d", b, r);
            x = resnet(E, x, H, W, cin, ch[b], pre);
            if (!x.ok()) return Tensor{};
            { char tg[64]; snprintf(tg, sizeof tg, "down%d_res%d", b, r); vae_dump(E, x, H, W, ch[b], tg); }
            cin = ch[b];
        }
        if (b < 3) {
            char wn[192], bn[192];
            snprintf(wn, sizeof wn, "vae.encoder.down_blocks.%d.downsamplers.0.conv.weight", b);
            snprintf(bn, sizeof bn, "vae.encoder.down_blocks.%d.downsamplers.0.conv.bias", b);
            // Downsample2D in the Flux2 VAE uses padding=0 (asymmetric pad (0,1,0,1)).
            x = conv2d(E, x, H, W, cin, wn, bn, 3, 3, 2, 0, 0, H / 2, W / 2);
            if (!x.ok()) return Tensor{};
            H /= 2;
            W /= 2;
            { char tg[64]; snprintf(tg, sizeof tg, "down%d_pool", b); vae_dump(E, x, H, W, cin, tg); }
        }
    }
    x = resnet(E, x, H, W, 512, 512, "vae.encoder.mid_block.resnets.0");
    if (!x.ok()) return Tensor{};
    vae_dump(E, x, H, W, 512, "mid_res0");
    if (vae_attention(E, x, H * W, 512, "vae.encoder.mid_block.attentions.0")) return Tensor{};
    vae_dump(E, x, H, W, 512, "mid_attn");
    x = resnet(E, x, H, W, 512, 512, "vae.encoder.mid_block.resnets.1");
    if (!x.ok()) return Tensor{};
    vae_dump(E, x, H, W, 512, "mid_res1");
    if (gn(E, x, H * W, 512, "vae.encoder.conv_norm_out.weight", "vae.encoder.conv_norm_out.bias",
           32, 1e-6f, 1)) return Tensor{};
    vae_dump(E, x, H, W, 512, "gn_out");
    x = conv2d(E, x, H, W, 512, "vae.encoder.conv_out.weight", "vae.encoder.conv_out.bias", 3, 3, 1, 1, 0);
    if (!x.ok()) return Tensor{};
    vae_dump(E, x, H, W, 64, "conv_out");
    x = conv2d(E, x, H, W, 64, "vae.quant_conv.weight", "vae.quant_conv.bias", 1, 1, 1, 0, 0);
    vae_dump(E, x, H, W, 64, "quant");
    return x;
}

Tensor vae_feature2(IdolEngine& E, Tensor enc, uint64_t seed) {
    const int TK = 4096, CT = 128;
    std::mt19937_64 rng(seed * 6364136223846793005ull + 1442695040888963407ull);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> eps((size_t)TK * CT);
    for (auto& v : eps) v = nd(rng);
    CUdeviceptr ep = E.pool.alloc(eps.size() * 4);
    E.drv.htod(ep, eps.data(), eps.size() * 4);
    Tensor packed = E.tm(TK, CT);
    if (!packed.ok()) return Tensor{};
    int H = 128, W = 128;
    CUdeviceptr mp = enc.p, lp = enc.p + 32 * 2;
    void* p[] = {&mp, &lp, &ep, &packed.p, &H, &W};
    if (E.launch("k_pack_latent", dim3((TK * CT + 255) / 256), dim3(256), 0, p)) return Tensor{};
    E.pool.release(ep, eps.size() * 4);
    E.rel(enc);
    CUdeviceptr rm = E.wget("vae.bn.running_mean");
    CUdeviceptr rv = E.wget("vae.bn.running_var");
    if (!rm || !rv) { fail("vae bn"); return Tensor{}; }
    Tensor normed = E.tm(TK, CT);
    if (!normed.ok()) return Tensor{};
    CUdeviceptr nz = 0;
    int HW = TK, C = CT;
    float beps = 1e-4f;
    void* q[] = {&packed.p, &nz, &nz, &rm, &rv, &normed.p, &HW, &C, &beps};
    if (E.launch("k_batchnorm", dim3((HW * C + 255) / 256), dim3(256), 0, q)) return Tensor{};
    E.rel(packed);
    Tensor f2 = E.tm(4101, CT);
    if (!f2.ok()) return Tensor{};
    E.zero(f2, (int64_t)f2.r * f2.c);
    CUdeviceptr dst = f2.p + (size_t)5 * CT * 2;
    int rows = TK, cols = CT;
    void* cr[] = {&dst, &f2.c, &normed.p, &normed.c, &rows, &cols};
    E.launch("k_copy_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, cr);
    E.rel(normed);
    return f2;
}

int repo_apply(IdolEngine& E, Tensor x, int rows, const char* prefix, Tensor& q, Tensor& k, int heads) {
    char b[384], b2[384];
    Tensor h = E.tm(rows, 1024);
    if (!h.ok()) return 1;
    snprintf(b, sizeof b, "%s.norm.weight", prefix);
    snprintf(b2, sizeof b2, "%s.norm.bias", prefix);
    if (E.ln_to(x, rows, 1024, b, b2, 1e-5f, 0, 0, h.p)) return 1;
    snprintf(b, sizeof b, "%s.gate_map.weight", prefix);
    Tensor g = E.gemm(h, b, nullptr, 3);
    snprintf(b, sizeof b, "%s.content_map.weight", prefix);
    Tensor c = E.gemm(h, b, nullptr, 0);
    E.rel(h);
    if (!g.ok() || !c.ok()) return 1;
    int n = g.r * g.c;
    void* m[] = {&g.p, &c.p, &n};
    E.launch("k_mul", dim3((n + 255) / 256), dim3(256), 0, m);
    E.rel(c);
    snprintf(b, sizeof b, "%s.final_map.weight", prefix);
    Tensor d = E.gemm(g, b, nullptr, 0);
    E.rel(g);
    if (!d.ok()) return 1;
    Tensor dc = E.tmx(d.r, 48);
    if (!dc.ok()) return 1;
    int zero = 0, rows2 = d.r, cols2 = 48;
    void* cp[] = {&dc.p, &dc.c, &zero, &d.p, &d.c, &rows2, &cols2};
    E.launch("k_copy_cols", dim3((rows2 * cols2 + 255) / 256), dim3(256), 0, cp);
    E.rel(d);
    CUdeviceptr df = E.pool.alloc((size_t)rows2 * 48 * 4);
    int s48 = 48;
    void* cv[] = {&dc.p, &s48, &df, &s48, &rows2, &cols2};
    E.launch("k_f16_to_f32_rows", dim3((rows2 * cols2 + 255) / 256), dim3(256), 0, cv);
    E.rel(dc);
    snprintf(b, sizeof b, "%s.freqs_0", prefix);
    CUdeviceptr f0 = E.wget(b);
    snprintf(b, sizeof b, "%s.freqs_1", prefix);
    CUdeviceptr f1 = E.wget(b);
    snprintf(b, sizeof b, "%s.freqs_2", prefix);
    CUdeviceptr f2 = E.wget(b);
    if (!f0 || !f1 || !f2) return fail("repo freqs");
    CUdeviceptr cosv = E.pool.alloc((size_t)rows2 * heads * 32 * 4);
    CUdeviceptr sinv = E.pool.alloc((size_t)rows2 * heads * 32 * 4);
    int rr = rows2, hh = heads;
    void* ra[] = {&df, &f0, &f1, &f2, &cosv, &sinv, &rr, &hh};
    if (E.launch("k_repo_angles", dim3((rr * hh + 255) / 256), dim3(256), 0, ra)) return 1;
    int np = rr * hh * 32;
    void* rq[] = {&q.p, &cosv, &sinv, &rr, &hh};
    void* rk[] = {&k.p, &cosv, &sinv, &rr, &hh};
    E.launch("k_rope_pairs", dim3((np + 255) / 256), dim3(256), 0, rq);
    E.launch("k_rope_pairs", dim3((np + 255) / 256), dim3(256), 0, rk);
    E.pool.release(df, (size_t)rows2 * 48 * 4);
    E.pool.release(cosv, (size_t)rows2 * heads * 32 * 4);
    E.pool.release(sinv, (size_t)rows2 * heads * 32 * 4);
    return 0;
}

int split_qkv(IdolEngine& E, Tensor qkv, int rows, Tensor& q, Tensor& k, Tensor& v) {
    q = E.tm(rows, 1024);
    k = E.tm(rows, 1024);
    v = E.tm(rows, 1024);
    if (!q.ok() || !k.ok() || !v.ok()) return 1;
    int zero = 0, cols = 1024, rw = rows;
    CUdeviceptr s0 = qkv.p, s1 = qkv.p + 1024 * 2, s2 = qkv.p + 2048 * 2;
    void* a[] = {&q.p, &q.c, &zero, &s0, &qkv.c, &rw, &cols};
    void* b[] = {&k.p, &k.c, &zero, &s1, &qkv.c, &rw, &cols};
    void* c[] = {&v.p, &v.c, &zero, &s2, &qkv.c, &rw, &cols};
    dim3 g((rw * cols + 255) / 256);
    E.launch("k_copy_cols", g, dim3(256), 0, a);
    E.launch("k_copy_cols", g, dim3(256), 0, b);
    E.launch("k_copy_cols", g, dim3(256), 0, c);
    return 0;
}
int set_mod(IdolEngine& E, Tensor mod, Tensor tmod, const char* stname) {
    int one = 1, n = 6144;
    void* cr[] = {&mod.p, &mod.c, &tmod.p, &tmod.c, &one, &n};
    if (E.launch("k_copy_rows", dim3((n + 255) / 256), dim3(256), 0, cr)) return 1;
    CUdeviceptr st = E.wget(stname);
    if (!st) return fail("missing %s", stname);
    void* ad[] = {&mod.p, &st, &n};
    return E.launch("k_add", dim3((n + 255) / 256), dim3(256), 0, ad);
}

static void den_dump(IdolEngine& E, Tensor t, int rows, const char* tag);
static int g_blk_count = 0;
static char g_blk_pfx[16] = "x";
#define BLKDUMP(t, r, nm) do { if (g_blk_count < 6) { char tg[64]; snprintf(tg, sizeof tg, "%s_%s", g_blk_pfx, nm); den_dump(E, t, r, tg); } } while (0)

int block_mod(IdolEngine& E, Tensor& x, int rows, int Lq, const char* pre, const char* rpre,
              CUdeviceptr mod) {
    snprintf(g_blk_pfx, sizeof g_blk_pfx, "mod%d", g_blk_count);
    ++g_blk_count;
    if (g_blk_count == 3) { Tensor md{mod, 1, 6144, 0}; den_dump(E, md, 1, "modvec"); }
    char b[384], b2[384];
    int cols = 1024;
    Tensor n1 = E.tm(rows, cols);
    if (!n1.ok()) return 1;
    if (E.ln_to(x, rows, cols, nullptr, nullptr, 1e-6f, mod + 1024 * 2, mod, n1.p)) return 1;
    BLKDUMP(n1, rows, "n1");
    snprintf(b, sizeof b, "%s.attn.qkv.weight", pre);
    snprintf(b2, sizeof b2, "%s.attn.qkv.bias", pre);
    Tensor qkv = E.gemm(n1, b, b2, 0);
    E.rel(n1);
    if (!qkv.ok()) return 1;
    BLKDUMP(qkv, rows, "qkv");
    Tensor q, k, v;
    if (split_qkv(E, qkv, rows, q, k, v)) return 1;
    E.rel(qkv);
    if (repo_apply(E, x, rows, rpre, q, k, 16)) return 1;
    BLKDUMP(q, rows, "q");
    BLKDUMP(v, rows, "v");
    snprintf(b, sizeof b, "%s.attn.q_norm.gamma", pre);
    if (E.rms(q, rows, 16, b)) return 1;
    snprintf(b, sizeof b, "%s.attn.k_norm.gamma", pre);
    if (E.rms(k, rows, 16, b)) return 1;
    BLKDUMP(q, rows, "q_rms");
    BLKDUMP(k, rows, "k_rms");
    Tensor o = E.flash(q, k, v, Lq, Lq, 16);
    BLKDUMP(o, rows, "o");
    E.rel(q);
    E.rel(k);
    E.rel(v);
    if (!o.ok()) return 1;
    snprintf(b, sizeof b, "%s.attn.out.weight", pre);
    snprintf(b2, sizeof b2, "%s.attn.out.bias", pre);
    Tensor o2 = E.gemm(o, b, b2, 0);
    E.rel(o);
    if (!o2.ok()) return 1;
    BLKDUMP(o2, rows, "o2");
    CUdeviceptr gate = mod + 2048 * 2;
    void* sa[] = {&x.p, &o2.p, &gate, &rows, &cols};
    E.launch("k_add_scaled", dim3((rows * cols + 255) / 256), dim3(256), 0, sa);
    E.rel(o2);
    Tensor n2 = E.tm(rows, cols);
    if (!n2.ok()) return 1;
    if (E.ln_to(x, rows, cols, nullptr, nullptr, 1e-6f, mod + 4096 * 2, mod + 3072 * 2, n2.p)) return 1;
    BLKDUMP(x, rows, "x");
    BLKDUMP(n2, rows, "n2");
    snprintf(b, sizeof b, "%s.mlp.mlp.0.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.0.bias", pre);
    Tensor g = E.gemm(n2, b, b2, 2);
    E.rel(n2);
    if (!g.ok()) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.2.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.2.bias", pre);
    BLKDUMP(g, rows, "g");
    Tensor d = E.gemm(g, b, b2, 0);
    E.rel(g);
    if (!d.ok()) return 1;
    BLKDUMP(d, rows, "d");
    gate = mod + 5120 * 2;
    void* sa2[] = {&x.p, &d.p, &gate, &rows, &cols};
    E.launch("k_add_scaled", dim3((rows * cols + 255) / 256), dim3(256), 0, sa2);
    E.rel(d);
    return 0;
}

int block_plain(IdolEngine& E, Tensor& x, int rows, int Lq, const char* pre, const char* rpre) {
    snprintf(g_blk_pfx, sizeof g_blk_pfx, "plain%d", g_blk_count++);
    char b[384], b2[384];
    int cols = 1024;
    Tensor n1 = E.tm(rows, cols);
    if (!n1.ok()) return 1;
    snprintf(b, sizeof b, "%s.norm1.weight", pre);
    snprintf(b2, sizeof b2, "%s.norm1.bias", pre);
    if (E.ln_to(x, rows, cols, b, b2, 1e-6f, 0, 0, n1.p)) return 1;
    BLKDUMP(n1, rows, "n1");
    snprintf(b, sizeof b, "%s.attn.qkv.weight", pre);
    snprintf(b2, sizeof b2, "%s.attn.qkv.bias", pre);
    Tensor qkv = E.gemm(n1, b, b2, 0);
    E.rel(n1);
    if (!qkv.ok()) return 1;
    BLKDUMP(qkv, rows, "qkv");
    Tensor q, k, v;
    if (split_qkv(E, qkv, rows, q, k, v)) return 1;
    E.rel(qkv);
    if (repo_apply(E, x, rows, rpre, q, k, 16)) return 1;
    BLKDUMP(q, rows, "q");
    BLKDUMP(v, rows, "v");
    snprintf(b, sizeof b, "%s.attn.q_norm.gamma", pre);
    if (E.rms(q, rows, 16, b)) return 1;
    snprintf(b, sizeof b, "%s.attn.k_norm.gamma", pre);
    if (E.rms(k, rows, 16, b)) return 1;
    BLKDUMP(q, rows, "q_rms");
    BLKDUMP(k, rows, "k_rms");
    Tensor o = E.flash(q, k, v, Lq, Lq, 16);
    BLKDUMP(o, rows, "o");
    E.rel(q);
    E.rel(k);
    E.rel(v);
    if (!o.ok()) return 1;
    snprintf(b, sizeof b, "%s.attn.out.weight", pre);
    snprintf(b2, sizeof b2, "%s.attn.out.bias", pre);
    Tensor o2 = E.gemm(o, b, b2, 0);
    E.rel(o);
    if (!o2.ok()) return 1;
    add_t(E, x, o2);
    E.rel(o2);
    Tensor n2 = E.tm(rows, cols);
    if (!n2.ok()) return 1;
    snprintf(b, sizeof b, "%s.norm2.weight", pre);
    snprintf(b2, sizeof b2, "%s.norm2.bias", pre);
    if (E.ln_to(x, rows, cols, b, b2, 1e-6f, 0, 0, n2.p)) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.0.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.0.bias", pre);
    Tensor g = E.gemm(n2, b, b2, 2);
    E.rel(n2);
    if (!g.ok()) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.2.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.2.bias", pre);
    Tensor d = E.gemm(g, b, b2, 0);
    E.rel(g);
    if (!d.ok()) return 1;
    add_t(E, x, d);
    E.rel(d);
    return 0;
}
static std::wstring g_den_dir;
static int g_den_layer = 0;

static void den_dump(IdolEngine& E, Tensor t, int rows, const char* tag) {
    if (g_den_dir.empty()) return;
    int cols = t.c;
    std::vector<float> h((size_t)rows * (size_t)cols, 0.f);
    if (download_f32(E, t, rows, cols, h.data())) { fprintf(stderr, "[den] %s dl fail\n", tag); return; }
    float mn = 1e30f, mx = -1e30f;
    double sum = 0.0;
    size_t bad = 0;
    for (float f : h) {
        if (!std::isfinite(f)) { ++bad; continue; }
        mn = std::min(mn, f); mx = std::max(mx, f); sum += f;
    }
    fprintf(stderr, "[den] %-16s min %.3f max %.3f mean %.4f bad %zu\n", tag, mn, mx, sum / (double)h.size(), bad);
    std::wstring wname(tag, tag + strlen(tag));
    std::wstring path = g_den_dir + L"/" + wname + L".bin";
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (f) { fwrite(h.data(), sizeof(float), h.size(), f); fclose(f); }
}

int denoise_forward(IdolEngine& E, Tensor f1, Tensor f2, Tensor latent, Tensor cam, float t,
                    Tensor& out_lat, Tensor& out_cam) {
    if (g_den_dir.empty()) {
        wchar_t buf[MAX_PATH];
        DWORD nb = GetEnvironmentVariableW(L"IDOL_DEBUG_DEN_DIR", buf, MAX_PATH);
        if (nb > 0 && nb < MAX_PATH) g_den_dir = buf;
        const char* lv = getenv("IDOL_DEBUG_DEN_LAYER");
        g_den_layer = lv ? atoi(lv) : 0;
    }
    char b[384];
    int cols = 1024, one = 1;
    float te[256];
    for (int i = 0; i < 128; ++i) {
        double fr = std::exp(-std::log(10000.0) * i / 128.0);
        double a = (double)t * fr;
        te[i] = (float)std::cos(a);
        te[128 + i] = (float)std::sin(a);
    }
    CUdeviceptr tef = E.pool.alloc(1024);
    E.drv.htod(tef, te, 1024);
    Tensor tsin = E.tm(1, 256);
    if (!tsin.ok()) return 1;
    E.zero(tsin, (int64_t)tsin.r * tsin.c);
    int n256 = 256;
    void* cv[] = {&tef, &n256, &tsin.p, &tsin.c, &one, &n256};
    E.launch("k_f32_to_f16_rows", dim3(1), dim3(256), 0, cv);
    E.pool.release(tef, 1024);
    Tensor temb = E.gemm(tsin, "denoiser.t_embedder.mlp.0.weight", "denoiser.t_embedder.mlp.0.bias", 3);
    E.rel(tsin);
    if (!temb.ok()) return 1;
    temb = E.gemm(temb, "denoiser.t_embedder.mlp.2.weight", "denoiser.t_embedder.mlp.2.bias", 0);
    if (!temb.ok()) return 1;
    Tensor temb_act = copy_t(E, temb);
    if (!temb_act.ok()) return 1;
    act_t(E, temb_act, 1, (int64_t)temb_act.r * temb_act.c);
    Tensor tmod = E.gemm(temb_act, "denoiser.adaLN_modulation.1.weight",
                         "denoiser.adaLN_modulation.1.bias", 0);
    E.rel(temb_act);
    if (!tmod.ok()) return 1;
    den_dump(E, tmod, 1, "tmod");
    den_dump(E, temb, 1, "temb");

    Tensor hc = E.gemm(f1, "denoiser.cond_embedder.weight", "denoiser.cond_embedder.bias", 0);
    Tensor hc2 = E.gemm(f2, "denoiser.cond_embedder2.weight", "denoiser.cond_embedder2.bias", 0);
    if (!hc.ok() || !hc2.ok()) return 1;
    add_t(E, hc, hc2);
    E.rel(hc2);
    E.zero_tail(hc, 4101);
    den_dump(E, hc, 4101, "cond");
    for (int i = 0; i < 2; ++i) {
        char pre[128], rpre[128];
        snprintf(pre, sizeof pre, "denoiser.context_refiner.%d", i);
        snprintf(rpre, sizeof rpre, "denoiser.context_repo_layers.%d", i);
        if (block_plain(E, hc, hc.r, 4101, pre, rpre)) return 1;
        { char tg[64]; snprintf(tg, sizeof tg, "ctx%d", i); den_dump(E, hc, 4101, tg); }
    }
    Tensor hx = E.gemm(latent, "denoiser.input_layer.weight", "denoiser.input_layer.bias", 0);
    if (!hx.ok()) return 1;
    CUdeviceptr pev = E.wget("denoiser.pos_embed");
    if (!pev) { fail("pos_embed"); return 1; }
    Tensor peT{pev, 8192, cols, 0};
    add_t(E, hx, peT);
    den_dump(E, hx, 8192, "hx_pos");
    Tensor mod = E.tm(1, 6144);
    if (!mod.ok()) return 1;
    for (int i = 0; i < 2; ++i) {
        char pre[128], rpre[128];
        snprintf(pre, sizeof pre, "denoiser.noise_refiner.%d", i);
        snprintf(rpre, sizeof rpre, "denoiser.noise_repo_layers.%d", i);
        snprintf(b, sizeof b, "%s.shift_table", pre);
        if (set_mod(E, mod, tmod, b)) return 1;
        if (block_mod(E, hx, hx.r, 8192, pre, rpre, mod.p)) return 1;
        { char tg[64]; snprintf(tg, sizeof tg, "nz%d", i); den_dump(E, hx, 8192, tg); }
    }
    Tensor hcam = E.gemm(cam, "denoiser.cam_refiner.mlp.0.weight", "denoiser.cam_refiner.mlp.0.bias", 2);
    if (!hcam.ok()) return 1;
    hcam = E.gemm(hcam, "denoiser.cam_refiner.mlp.2.weight", "denoiser.cam_refiner.mlp.2.bias", 0);
    if (!hcam.ok()) return 1;

    const int LJ = 8192 + 4101 + 1;
    Tensor joint = E.tm(LJ, cols);
    if (!joint.ok()) return 1;
    int rw = 8192;
    void* j1[] = {&joint.p, &joint.c, &hx.p, &hx.c, &rw, &cols};
    E.launch("k_copy_rows", dim3((rw * cols + 255) / 256), dim3(256), 0, j1);
    CUdeviceptr jo = joint.p + (size_t)8192 * cols * 2;
    rw = 4101;
    void* j2[] = {&jo, &joint.c, &hc.p, &hc.c, &rw, &cols};
    E.launch("k_copy_rows", dim3((rw * cols + 255) / 256), dim3(256), 0, j2);
    jo = joint.p + (size_t)12293 * cols * 2;
    rw = 1;
    void* j3[] = {&jo, &joint.c, &hcam.p, &hcam.c, &rw, &cols};
    E.launch("k_copy_rows", dim3((rw * cols + 255) / 256), dim3(256), 0, j3);
    E.rel(hx);
    E.rel(hc);
    E.rel(hcam);
    E.zero_tail(joint, LJ);
    den_dump(E, joint, LJ, "joint");
    for (int i = 0; i < 24; ++i) {
        char pre[128], rpre[128];
        snprintf(pre, sizeof pre, "denoiser.blocks.%d", i);
        snprintf(rpre, sizeof rpre, "denoiser.repo_layers.%d", i);
        snprintf(b, sizeof b, "%s.shift_table", pre);
        if (set_mod(E, mod, tmod, b)) return 1;
        if (block_mod(E, joint, joint.r, LJ, pre, rpre, mod.p)) return 1;
        if (i < 3 || i == g_den_layer) { char tg[64]; snprintf(tg, sizeof tg, "blk%d", i); den_dump(E, joint, LJ, tg); }
        if (i == 0) g_blk_count = 100000;
    }
    E.rel(mod);
    E.rel(tmod);
    if (E.ln(joint, 8192, cols, nullptr, nullptr, 1e-5f)) return 1;
    den_dump(E, joint, 8192, "pj_ln");
    Tensor fmod = E.tm(1, 2048);
    if (!fmod.ok()) return 1;
    E.zero(fmod, (int64_t)fmod.r * fmod.c);
    CUdeviceptr sft = fmod.p, scl = fmod.p + 1024 * 2;
    int c1024 = 1024;
    void* s1[] = {&sft, &fmod.c, &temb.p, &temb.c, &one, &c1024};
    void* s2[] = {&scl, &fmod.c, &temb.p, &temb.c, &one, &c1024};
    E.launch("k_copy_rows", dim3((c1024 + 255) / 256), dim3(256), 0, s1);
    E.launch("k_copy_rows", dim3((c1024 + 255) / 256), dim3(256), 0, s2);
    E.rel(temb);
    CUdeviceptr st = E.wget("denoiser.shift_table");
    if (!st) { fail("shift_table"); return 1; }
    int n2048 = 2048;
    void* ad[] = {&fmod.p, &st, &n2048};
    E.launch("k_add", dim3((n2048 + 255) / 256), dim3(256), 0, ad);
    int r8192 = 8192;
    void* ss[] = {&joint.p, &scl, &sft, &r8192, &cols};
    E.launch("k_scale_shift", dim3((r8192 * cols + 255) / 256), dim3(256), 0, ss);
    den_dump(E, fmod, 1, "pfmod");
    den_dump(E, joint, 8192, "pj_mod");
    Tensor hxview{joint.p, 8192, cols, 0};
    out_lat = E.gemm(hxview, "denoiser.out_layer.weight", "denoiser.out_layer.bias", 0);
    if (!out_lat.ok()) return 1;
    den_dump(E, out_lat, 8192, "out_lat");
    Tensor c1row = E.tm(1, cols);
    if (!c1row.ok()) return 1;
    E.zero(c1row, (int64_t)c1row.r * c1row.c);
    CUdeviceptr csrc = joint.p + (size_t)12293 * cols * 2, cdst = c1row.p;
    void* cc[] = {&cdst, &c1row.c, &csrc, &cols, &one, &cols};
    E.launch("k_copy_rows", dim3((cols + 255) / 256), dim3(256), 0, cc);
    if (E.ln_to(c1row, 1, cols, nullptr, nullptr, 1e-5f, 0, 0, c1row.p)) return 1;
    void* ss2[] = {&c1row.p, &scl, &sft, &one, &cols};
    E.launch("k_scale_shift", dim3((cols + 255) / 256), dim3(256), 0, ss2);
    out_cam = E.gemm(c1row, "denoiser.cam_out_layer.weight", "denoiser.cam_out_layer.bias", 0);
    E.rel(c1row);
    E.rel(joint);
    E.rel(fmod);
    return out_cam.ok() ? 0 : 1;
}
int systematic_row(const float* logits, int count, std::mt19937_64& rng, int* out_counts) {
    float mx = logits[0];
    for (int i = 1; i < 8; ++i) mx = std::fmax(mx, logits[i]);
    double p[8], cdf[8], acc = 0.0, sum = 0.0;
    for (int i = 0; i < 8; ++i) { p[i] = std::exp((double)logits[i] - (double)mx); sum += p[i]; }
    if (!(sum > 0.0)) {
        for (int i = 0; i < 8; ++i) p[i] = 0.125;
    } else {
        for (int i = 0; i < 8; ++i) p[i] /= sum;
    }
    for (int i = 0; i < 8; ++i) {
        acc += p[i];
        cdf[i] = acc < 1.0 - 1e-12 ? acc : 1.0 - 1e-12;
        out_counts[i] = 0;
    }
    if (count <= 0) return 0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double u0 = u01(rng) / (double)count;
    for (int k = 0; k < count; ++k) {
        double u = u0 + (double)k / (double)count;
        if (u > 1.0 - 1e-12) u = 1.0 - 1e-12;
        int idx = 0;
        while (idx < 7 && cdf[idx] <= u) ++idx;
        out_counts[idx]++;
    }
    return 0;
}

int dec_block(IdolEngine& E, Tensor& x, int rows, const char* pre, CUdeviceptr mod, Tensor latent);
int gs_block(IdolEngine& E, Tensor& x, int N, const char* pre, Tensor latent);

Tensor dec_forward(IdolEngine& E, Tensor pt, CUdeviceptr posf, int N, int lvl, Tensor latent,
                   CUdeviceptr freqs) {
    int c1024 = 1024, n = N, one = 1;
    Tensor h = E.gemm(pt, "decoder.in_proj.weight", "decoder.in_proj.bias", 0);
    if (!h.ok()) return Tensor{};
    CUdeviceptr pf = E.pool.alloc((size_t)N * 1024 * 4);
    if (!pf) return Tensor{};
    void* pe1[] = {&posf, &freqs, &pf, &n, &c1024};
    if (E.launch("k_pos_embed", dim3((N * 1024 + 255) / 256), dim3(256), 0, pe1)) return Tensor{};
    Tensor pe16 = E.tm(N, 1024);
    if (!pe16.ok()) return Tensor{};
    E.zero(pe16, (int64_t)pe16.r * pe16.c);
    void* pe2[] = {&pf, &c1024, &pe16.p, &pe16.c, &n, &c1024};
    E.launch("k_f32_to_f16_rows", dim3((N * 1024 + 255) / 256), dim3(256), 0, pe2);
    E.pool.release(pf, (size_t)N * 1024 * 4);
    add_t(E, h, pe16);
    E.rel(pe16);
    h = E.gemm(h, "decoder.input_layer.weight", "decoder.input_layer.bias", 0);
    if (!h.ok()) return Tensor{};
    float le[256];
    for (int i = 0; i < 128; ++i) {
        double fr = std::exp(-std::log(1024.0) * i / 128.0);
        double a = (double)lvl * fr * 2.0 * 3.14159265358979323846;
        le[i] = (float)std::cos(a);
        le[128 + i] = (float)std::sin(a);
    }
    CUdeviceptr lef = E.pool.alloc(1024);
    E.drv.htod(lef, le, 1024);
    Tensor lv16 = E.tm(1, 256);
    if (!lv16.ok()) return Tensor{};
    E.zero(lv16, (int64_t)lv16.r * lv16.c);
    int n256 = 256;
    void* lc[] = {&lef, &n256, &lv16.p, &lv16.c, &one, &n256};
    E.launch("k_f32_to_f16_rows", dim3(1), dim3(256), 0, lc);
    E.pool.release(lef, 1024);
    Tensor lemb = E.gemm(lv16, "decoder.l_embedder.mlp.0.weight", "decoder.l_embedder.mlp.0.bias", 3);
    E.rel(lv16);
    if (!lemb.ok()) return Tensor{};
    lemb = E.gemm(lemb, "decoder.l_embedder.mlp.2.weight", "decoder.l_embedder.mlp.2.bias", 0);
    if (!lemb.ok()) return Tensor{};
    act_t(E, lemb, 1, (int64_t)lemb.r * lemb.c);
    Tensor lmod = E.gemm(lemb, "decoder.adaLN_modulation.1.weight", "decoder.adaLN_modulation.1.bias", 0);
    E.rel(lemb);
    if (!lmod.ok()) return Tensor{};
    for (int i = 0; i < 4; ++i) {
        char pre[128];
        snprintf(pre, sizeof pre, "decoder.blocks.%d", i);
        if (dec_block(E, h, N, pre, lmod.p, latent)) return Tensor{};
    }
    E.rel(lmod);
    if (E.ln(h, N, c1024, nullptr, nullptr, 1e-5f)) return Tensor{};
    return E.gemm(h, "decoder.out_proj.weight", "decoder.out_proj.bias", 0);
}

int run_octree(IdolEngine& E, Tensor latent, int num_points, int level, uint64_t seed,
               std::vector<float>& out) {
    int c1024 = 1024, c3 = 3;
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int FN = 170;
    float freqs[FN];
    for (int i = 0; i < FN; ++i) freqs[i] = (float)std::pow(2.0, 10.0 * i / (FN - 1.0));
    CUdeviceptr fq = E.pool.alloc(FN * 4);
    E.drv.htod(fq, freqs, FN * 4);
    std::vector<int> coords(3, 0), counts(1, num_points);
    int N = 1;
    for (int l = 1; l <= level; ++l) {
        float res = (float)(1 << (l - 1));
        std::vector<float> pos((size_t)N * 3);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < 3; ++j) pos[(size_t)i * 3 + j] = ((float)coords[(size_t)i * 3 + j] + 0.5f) / res;
        CUdeviceptr pf = E.pool.alloc(pos.size() * 4);
        if (!pf) return 1;
        E.drv.htod(pf, pos.data(), pos.size() * 4);
        Tensor pt = E.tm(N, 3);
        if (!pt.ok()) return 1;
        E.zero(pt, (int64_t)pt.r * pt.c);
        int rN = N;
        void* cp[] = {&pf, &c3, &pt.p, &pt.c, &rN, &c3};
        E.launch("k_f32_to_f16_rows", dim3((N * 3 + 255) / 256), dim3(256), 0, cp);
        Tensor logits = dec_forward(E, pt, pf, N, 1 << l, latent, fq);
        E.pool.release(pf, pos.size() * 4);
        E.rel(pt);
        if (!logits.ok()) return 1;
        std::vector<float> all((size_t)N * 64);
        CUdeviceptr buf = E.pool.alloc(all.size() * 4);
        int c64 = 64;
        void* cv[] = {&logits.p, &logits.c, &buf, &c64, &rN, &c64};
        E.launch("k_f16_to_f32_rows", dim3((N * 64 + 255) / 256), dim3(256), 0, cv);
        E.rel(logits);
        if (E.sync()) return 1;
        E.drv.dtoh(all.data(), buf, all.size() * 4);
        E.pool.release(buf, all.size() * 4);
        std::vector<int> ncoords;
        std::vector<int> ncounts;
        int bins[8];
        for (int i = 0; i < N; ++i) {
            float probs[8];
            for (int j = 0; j < 8; ++j) probs[j] = all[(size_t)i * 64 + j];
            systematic_row(probs, counts[i], rng, bins);
            for (int b = 0; b < 8; ++b) {
                if (bins[b] == 0) continue;
                ncoords.push_back(coords[(size_t)i * 3 + 0] * 2 + (b & 1));
                ncoords.push_back(coords[(size_t)i * 3 + 1] * 2 + ((b >> 1) & 1));
                ncoords.push_back(coords[(size_t)i * 3 + 2] * 2 + ((b >> 2) & 1));
                ncounts.push_back(bins[b]);
            }
        }
        coords.swap(ncoords);
        counts.swap(ncounts);
        N = (int)counts.size();
        if (N == 0) return fail("octree produced no cells");
    }
    out.clear();
    out.reserve((size_t)num_points * 3);
    const float inv = 1.0f / (float)(1 << level);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < counts[i]; ++k) {
            out.push_back(((float)coords[(size_t)i * 3 + 0] + u01(rng)) * inv);
            out.push_back(((float)coords[(size_t)i * 3 + 1] + u01(rng)) * inv);
            out.push_back(((float)coords[(size_t)i * 3 + 2] + u01(rng)) * inv);
        }
    }
    E.pool.release(fq, FN * 4);
    return 0;
}
int dec_block(IdolEngine& E, Tensor& x, int rows, const char* pre, CUdeviceptr mod, Tensor latent) {
    char b[384], b2[384];
    int cols = 1024, zero = 0, c1024 = 1024, r8 = 8192;
    Tensor n1 = E.tm(rows, cols);
    if (!n1.ok()) return 1;
    if (E.ln_to(x, rows, cols, nullptr, nullptr, 1e-6f, mod + 1024 * 2, mod, n1.p)) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_q.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_q.bias", pre);
    Tensor q = E.gemm(n1, b, b2, 0);
    E.rel(n1);
    if (!q.ok()) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_kv.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_kv.bias", pre);
    Tensor kv = E.gemm(latent, b, b2, 0);
    if (!kv.ok()) return 1;
    Tensor k = E.tm(r8, cols), v = E.tm(r8, cols);
    if (!k.ok() || !v.ok()) return 1;
    CUdeviceptr s0 = kv.p, s1 = kv.p + 1024 * 2;
    void* ka[] = {&k.p, &k.c, &zero, &s0, &kv.c, &r8, &c1024};
    void* va[] = {&v.p, &v.c, &zero, &s1, &kv.c, &r8, &c1024};
    E.launch("k_copy_cols", dim3((r8 * c1024 + 255) / 256), dim3(256), 0, ka);
    E.launch("k_copy_cols", dim3((r8 * c1024 + 255) / 256), dim3(256), 0, va);
    E.rel(kv);
    snprintf(b, sizeof b, "%s.cross_attn.q_rms_norm.gamma", pre);
    if (E.rms(q, rows, 16, b)) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.k_rms_norm.gamma", pre);
    if (E.rms(k, r8, 16, b)) return 1;
    Tensor o = E.flash(q, k, v, rows, r8, 16);
    E.rel(q);
    E.rel(k);
    E.rel(v);
    if (!o.ok()) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_out.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_out.bias", pre);
    Tensor o2 = E.gemm(o, b, b2, 0);
    E.rel(o);
    if (!o2.ok()) return 1;
    BLKDUMP(o2, rows, "o2");
    CUdeviceptr gate = mod + 2048 * 2;
    void* sa[] = {&x.p, &o2.p, &gate, &rows, &cols};
    E.launch("k_add_scaled", dim3((rows * cols + 255) / 256), dim3(256), 0, sa);
    E.rel(o2);
    Tensor n2 = E.tm(rows, cols);
    if (!n2.ok()) return 1;
    if (E.ln_to(x, rows, cols, nullptr, nullptr, 1e-6f, mod + 4096 * 2, mod + 3072 * 2, n2.p)) return 1;
    BLKDUMP(x, rows, "x");
    BLKDUMP(n2, rows, "n2");
    snprintf(b, sizeof b, "%s.mlp.mlp.0.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.0.bias", pre);
    Tensor g = E.gemm(n2, b, b2, 2);
    E.rel(n2);
    if (!g.ok()) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.2.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.2.bias", pre);
    BLKDUMP(g, rows, "g");
    Tensor d = E.gemm(g, b, b2, 0);
    E.rel(g);
    if (!d.ok()) return 1;
    BLKDUMP(d, rows, "d");
    gate = mod + 5120 * 2;
    void* sa2[] = {&x.p, &d.p, &gate, &rows, &cols};
    E.launch("k_add_scaled", dim3((rows * cols + 255) / 256), dim3(256), 0, sa2);
    E.rel(d);
    return 0;
}

Tensor gs_forward(IdolEngine& E, Tensor pt, CUdeviceptr posf, int N, Tensor latent, CUdeviceptr freqs) {
    int c1024 = 1024, n = N;
    Tensor h = E.gemm(pt, "decoder_gs.in_proj.weight", "decoder_gs.in_proj.bias", 0);
    if (!h.ok()) return Tensor{};
    CUdeviceptr pf = E.pool.alloc((size_t)N * 1024 * 4);
    if (!pf) return Tensor{};
    void* pe1[] = {&posf, &freqs, &pf, &n, &c1024};
    if (E.launch("k_pos_embed", dim3((N * 1024 + 255) / 256), dim3(256), 0, pe1)) return Tensor{};
    Tensor pe16 = E.tm(N, 1024);
    if (!pe16.ok()) return Tensor{};
    E.zero(pe16, (int64_t)pe16.r * pe16.c);
    void* pe2[] = {&pf, &c1024, &pe16.p, &pe16.c, &n, &c1024};
    E.launch("k_f32_to_f16_rows", dim3((N * 1024 + 255) / 256), dim3(256), 0, pe2);
    E.pool.release(pf, (size_t)N * 1024 * 4);
    add_t(E, h, pe16);
    E.rel(pe16);
    h = E.gemm(h, "decoder_gs.input_layer.weight", "decoder_gs.input_layer.bias", 0);
    if (!h.ok()) return Tensor{};
    for (int i = 0; i < 16; ++i) {
        char pre[128];
        snprintf(pre, sizeof pre, "decoder_gs.blocks.%d", i);
        if (gs_block(E, h, N, pre, latent)) return Tensor{};
    }
    if (E.ln(h, N, c1024, nullptr, nullptr, 1e-5f)) return Tensor{};
    return E.gemm(h, "decoder_gs.out_proj.weight", "decoder_gs.out_proj.bias", 0);
}

int gs_block(IdolEngine& E, Tensor& x, int N, const char* pre, Tensor latent) {
    char b[384], b2[384];
    int cols = 1024, zero = 0, c1024 = 1024, r8 = 8192;
    Tensor n1 = E.tm(x.r, cols);
    if (!n1.ok()) return 1;
    if (E.ln_to(x, x.r, cols, nullptr, nullptr, 1e-6f, 0, 0, n1.p)) return 1;
    snprintf(b, sizeof b, "%s.self_attn.to_qkv.weight", pre);
    snprintf(b2, sizeof b2, "%s.self_attn.to_qkv.bias", pre);
    Tensor qkv = E.gemm(n1, b, b2, 0);
    E.rel(n1);
    if (!qkv.ok()) return 1;
    Tensor q, k, v;
    if (split_qkv(E, qkv, x.r, q, k, v)) return 1;
    E.rel(qkv);
    snprintf(b, sizeof b, "%s.self_attn.q_rms_norm.gamma", pre);
    if (E.rms(q, x.r, 16, b)) return 1;
    snprintf(b, sizeof b, "%s.self_attn.k_rms_norm.gamma", pre);
    if (E.rms(k, x.r, 16, b)) return 1;
    Tensor o = E.flash(q, k, v, N, N, 16);
    E.rel(q);
    E.rel(k);
    E.rel(v);
    if (!o.ok()) return 1;
    snprintf(b, sizeof b, "%s.self_attn.to_out.weight", pre);
    snprintf(b2, sizeof b2, "%s.self_attn.to_out.bias", pre);
    Tensor o2 = E.gemm(o, b, b2, 0);
    E.rel(o);
    if (!o2.ok()) return 1;
    add_t(E, x, o2);
    E.rel(o2);

    Tensor n2 = E.tm(x.r, cols);
    if (!n2.ok()) return 1;
    snprintf(b, sizeof b, "%s.norm2.weight", pre);
    snprintf(b2, sizeof b2, "%s.norm2.bias", pre);
    if (E.ln_to(x, x.r, cols, b, b2, 1e-6f, 0, 0, n2.p)) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_q.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_q.bias", pre);
    Tensor cq = E.gemm(n2, b, b2, 0);
    E.rel(n2);
    if (!cq.ok()) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_kv.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_kv.bias", pre);
    Tensor kv = E.gemm(latent, b, b2, 0);
    if (!kv.ok()) return 1;
    Tensor ck = E.tm(r8, cols), cv = E.tm(r8, cols);
    if (!ck.ok() || !cv.ok()) return 1;
    CUdeviceptr s0 = kv.p, s1 = kv.p + 1024 * 2;
    void* ka[] = {&ck.p, &ck.c, &zero, &s0, &kv.c, &r8, &c1024};
    void* va[] = {&cv.p, &cv.c, &zero, &s1, &kv.c, &r8, &c1024};
    E.launch("k_copy_cols", dim3((r8 * c1024 + 255) / 256), dim3(256), 0, ka);
    E.launch("k_copy_cols", dim3((r8 * c1024 + 255) / 256), dim3(256), 0, va);
    E.rel(kv);
    snprintf(b, sizeof b, "%s.cross_attn.q_rms_norm.gamma", pre);
    if (E.rms(cq, x.r, 16, b)) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.k_rms_norm.gamma", pre);
    if (E.rms(ck, r8, 16, b)) return 1;
    Tensor co = E.flash(cq, ck, cv, N, r8, 16);
    E.rel(cq);
    E.rel(ck);
    E.rel(cv);
    if (!co.ok()) return 1;
    snprintf(b, sizeof b, "%s.cross_attn.to_out.weight", pre);
    snprintf(b2, sizeof b2, "%s.cross_attn.to_out.bias", pre);
    Tensor co2 = E.gemm(co, b, b2, 0);
    E.rel(co);
    if (!co2.ok()) return 1;
    add_t(E, x, co2);
    E.rel(co2);

    Tensor n3 = E.tm(x.r, cols);
    if (!n3.ok()) return 1;
    if (E.ln_to(x, x.r, cols, nullptr, nullptr, 1e-6f, 0, 0, n3.p)) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.0.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.0.bias", pre);
    Tensor g = E.gemm(n3, b, b2, 2);
    E.rel(n3);
    if (!g.ok()) return 1;
    snprintf(b, sizeof b, "%s.mlp.mlp.2.weight", pre);
    snprintf(b2, sizeof b2, "%s.mlp.mlp.2.bias", pre);
    Tensor d = E.gemm(g, b, b2, 0);
    E.rel(g);
    if (!d.ok()) return 1;
    add_t(E, x, d);
    E.rel(d);
    return 0;
}

}  // namespace

namespace {

struct CtxGuard {
    CUcontext ctx;
    explicit CtxGuard(CUcontext c) : ctx(c) { if (ctx) cuCtxPushCurrent(ctx); }
    ~CtxGuard() { if (ctx) cuCtxPopCurrent(nullptr); }
};

inline float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = 0;
            while (!(mant & 0x400u)) { mant <<= 1; ++e; }
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

Tensor upload_f16(IdolEngine& E, int rows, int cols, const float* src, int sstride) {
    Tensor t = E.tm(rows, cols);
    if (!t.ok()) return t;
    E.zero(t, (int64_t)t.r * t.c);
    if (!src) return t;
    size_t bytes = (size_t)rows * cols * 4;
    CUdeviceptr host = E.pool.alloc(bytes);
    if (!host) { E.rel(t); return Tensor{}; }
    if (sstride == cols) {
        if (E.drv.htod(host, src, bytes) != CUDA_SUCCESS) {
            E.pool.release(host, bytes);
            E.rel(t);
            return Tensor{};
        }
    } else {
        std::vector<float> packed(bytes / 4);
        for (int r = 0; r < rows; ++r)
            memcpy(packed.data() + (size_t)r * cols, src + (size_t)r * sstride,
                   (size_t)cols * sizeof(float));
        if (E.drv.htod(host, packed.data(), bytes) != CUDA_SUCCESS) {
            E.pool.release(host, bytes);
            E.rel(t);
            return Tensor{};
        }
    }
    int r = rows, c = cols;
    void* a[] = {&host, &c, &t.p, &t.c, &r, &c};
    int rc = E.launch("k_f32_to_f16_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, a);
    E.pool.release(host, bytes);
    if (rc) { E.rel(t); return Tensor{}; }
    return t;
}

int download_f32(IdolEngine& E, Tensor t, int rows, int cols, float* dst) {
    size_t bytes = (size_t)rows * cols * 4;
    CUdeviceptr out = E.pool.alloc(bytes);
    if (!out) return fail("oom download");
    int r = rows, c = cols;
    void* a[] = {&t.p, &t.c, &out, &cols, &r, &c};
    int rc = E.launch("k_f16_to_f32_rows", dim3((rows * cols + 255) / 256), dim3(256), 0, a);
    if (!rc && E.drv.dtoh(dst, out, bytes) != CUDA_SUCCESS) rc = fail("dtoh failed");
    E.pool.release(out, bytes);
    if (rc) return 1;
    return E.sync();
}

// ---------------------------------------------------------------- host helpers
inline float inv_softplus(float x) { return x + std::log(-std::expm1(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplusf(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

// Rescales the sampling progress (0..100) into the slice of the overall job it belongs to.
struct StepProgress {
    IdolProgress cb;
    void* user;
    int lo, hi;
};

void step_progress(int percent, const char* message, void* user) {
    auto* s = static_cast<StepProgress*>(user);
    if (!s->cb) return;
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    s->cb(s->lo + (s->hi - s->lo) * percent / 100, message, s->user);
}

void resize_planar(const float* src, int sw, int sh, int sc, float* dst, int dw, int dh, int dc) {
    for (int y = 0; y < dh; ++y) {
        float sy = dh > 1 ? (float)y * (float)(sh - 1) / (float)(dh - 1) : 0.f;
        int y0 = (int)sy;
        if (y0 > sh - 1) y0 = sh - 1;
        int y1 = y0 + 1 < sh ? y0 + 1 : y0;
        float fy = sy - (float)y0;
        for (int x = 0; x < dw; ++x) {
            float sx = dw > 1 ? (float)x * (float)(sw - 1) / (float)(dw - 1) : 0.f;
            int x0 = (int)sx;
            if (x0 > sw - 1) x0 = sw - 1;
            int x1 = x0 + 1 < sw ? x0 + 1 : x0;
            float fx = sx - (float)x0;
            for (int c = 0; c < dc; ++c) {
                float v00 = src[((size_t)y0 * sw + x0) * sc + c];
                float v01 = src[((size_t)y0 * sw + x1) * sc + c];
                float v10 = src[((size_t)y1 * sw + x0) * sc + c];
                float v11 = src[((size_t)y1 * sw + x1) * sc + c];
                dst[((size_t)y * dw + x) * dc + c] =
                    (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
            }
        }
    }
}

int encode_dino(IdolEngine& E, const float* prepared, float* feature1);
int encode_vae(IdolEngine& E, const float* prepared, float* feature2);

int encode_image(IdolEngine& E, const float* prepared, float* feature1, float* feature2) {
    if (encode_dino(E, prepared, feature1)) return 1;
    return encode_vae(E, prepared, feature2);
}

int encode_dino(IdolEngine& E, const float* prepared, float* feature1) {
    Tensor tok = dino_encode(E, prepared);
    if (!tok.ok()) return 1;
    if (download_f32(E, tok, 4101, 1280, feature1)) return fail("dino download");
    E.rel(tok);
    E.free_weights();
    E.pool.clear();
    E.pool.peak = 0;
    return 0;
}

int encode_vae(IdolEngine& E, const float* prepared, float* feature2) {
    Tensor enc = vae_encode(E, prepared);
    if (!enc.ok()) return 1;
    Tensor f2 = vae_feature2(E, enc, 0);
    if (!f2.ok()) return 1;
    if (download_f32(E, f2, 4101, 128, feature2)) return 1;
    E.rel(f2);
    E.free_weights();
    E.pool.clear();
    E.pool.peak = 0;
    return 0;
}

int run_velocity(IdolEngine& E, Tensor f1, Tensor f2, Tensor latent, Tensor cam, float t, Tensor& vlat,
                 Tensor& vcam) {
    Tensor zlat, zcam;
    if (denoise_forward(E, f1, f2, latent, cam, t, zlat, zcam)) return 1;
    vlat = zlat;
    vcam = zcam;
    return 0;
}

void fill_normal(float* dst, size_t n, uint64_t& state) {
    for (size_t i = 0; i < n; i += 2) {
        uint64_t a = state += 0x9E3779B97F4A7C15ull;
        uint64_t b = state += 0x9E3779B97F4A7C15ull;
        auto u = [](uint64_t x) {
            x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
            x ^= x >> 27; x *= 0x94D049BB133111EBull;
            x ^= x >> 31;
            return (float)((x >> 11) * (1.0 / 9007199254740992.0));
        };
        float u1 = 1.0f - u(a), u2 = u(b);
        float r = std::sqrt(-2.0f * std::log(u1 > 1e-7f ? u1 : 1e-7f));
        dst[i] = r * std::cos(6.283185307179586f * u2);
        if (i + 1 < n) dst[i + 1] = r * std::sin(6.283185307179586f * u2);
    }
}

}  // namespace

extern "C" {

uint32_t idol_engine_abi(void) { return kAbi; }

const char* idol_engine_last_error(void) { return g_error; }

int32_t idol_engine_create(int32_t device, const char* weights_path, IdolEngine** out) {
    if (!out) return fail("out is null");
    *out = nullptr;
    g_error[0] = '\0';
    auto* e = new IdolEngine();
    g_debug_sync = GetEnvironmentVariableA("IDOL_DEBUG_SYNC", nullptr, 0) > 0;
    g_debug_pool = GetEnvironmentVariableA("IDOL_DEBUG_POOL", nullptr, 0) > 0;
    g_profile = GetEnvironmentVariableA("IDOL_PROFILE", nullptr, 0) > 0;
    { LARGE_INTEGER fq; QueryPerformanceFrequency(&fq); g_qpc_freq = (double)fq.QuadPart; }
    if (!e->drv.load()) { delete e; return fail("cannot load nvcuda.dll (NVIDIA driver missing)"); }
    if (e->drv.init(0) != CUDA_SUCCESS) { delete e; return fail("cuInit failed"); }
    CUdevice dev = 0;
    if (e->drv.device_get(&dev, device) != CUDA_SUCCESS) { delete e; return fail("no CUDA device %d", device); }
    e->drv.device_name(e->dev_name, (int)sizeof(e->dev_name), dev);
    e->drv.device_attr(&e->cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    e->drv.device_attr(&e->cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (e->drv.ctx_create(&e->ctx, 0, dev) != CUDA_SUCCESS) { delete e; return fail("context create failed"); }
    {
        uint64_t f = 0, t = 0;
        if (cuMemGetInfo(&f, &t) == CUDA_SUCCESS) e->base_bytes = t - f;
    }
    if (!e->weights.open(weights_path)) { delete e; return fail("cannot open weights %s", weights_path); }
    CUresult r = e->drv.module_load(&e->mod, kIdolPtx, 0, nullptr, nullptr);
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        e->drv.err_string(r, &s);
        delete e;
        return fail("PTX load failed: %s", s ? s : "?");
    }
    *out = e;
    return 0;
}

void idol_engine_destroy(IdolEngine* engine) { delete engine; }

int32_t idol_engine_info(IdolEngine* engine, char* name, int name_len, int32_t* major, int32_t* minor,
                         uint64_t* free_bytes, uint64_t* total_bytes) {
    if (!engine) return fail("engine is null");
    CtxGuard guard(engine->ctx);
    if (name && name_len > 0) {
        strncpy_s(name, (size_t)name_len, engine->dev_name, _TRUNCATE);
    }
    if (major) *major = engine->cc_major;
    if (minor) *minor = engine->cc_minor;
    if (free_bytes || total_bytes) {
        uint64_t f = 0, t = 0;
        cuMemGetInfo(&f, &t);
        if (free_bytes) *free_bytes = f;
        if (total_bytes) *total_bytes = t;
    }
    return 0;
}

int32_t idol_memory(IdolEngine* engine, uint64_t* used, uint64_t* peak) {
    if (!engine) return fail("engine is null");
    if (used) *used = g_dev_bytes;
    if (peak) *peak = g_dev_peak;
    return 0;
}

int32_t idol_engine_set_remove_background(IdolEngine* engine, int32_t enabled) {
    if (!engine) return fail("idol_engine_set_remove_background: null engine");
    engine->remove_background = enabled ? 1 : 0;
    return 0;
}

int32_t idol_engine_reset_peak(IdolEngine* engine) {
    if (!engine) return fail("engine is null");
    g_dev_peak = g_dev_bytes;
    engine->peak_bytes = 0;
    engine->pool.peak = 0;
    return 0;
}

int32_t idol_engine_set_crop_margin(IdolEngine* engine, float margin) {
    if (!engine) return fail("engine is null");
    if (!(margin > 1.0f) || margin > 4.0f) return fail("crop margin out of range");
    engine->crop_margin = margin;
    return 0;
}

int32_t idol_engine_unload(IdolEngine* engine) {
    if (!engine) return fail("engine is null");
    CtxGuard guard(engine->ctx);
    cuCtxSynchronize();
    engine->free_weights();
    engine->pool.clear();
    engine->kernels.clear();
    uint64_t f = 0, t = 0;
    if (cuMemGetInfo(&f, &t) == CUDA_SUCCESS) {
        uint64_t used = t - f;
        if (used > engine->peak_bytes) engine->peak_bytes = used;
    }
    cuCtxSynchronize();
    return 0;
}

int32_t idol_encode(IdolEngine* engine, const float* prepared, float* feature1, float* feature2) {
    if (!engine || !prepared || !feature1 || !feature2) return fail("idol_encode: null argument");
    CtxGuard guard(engine->ctx);
    if (encode_image(*engine, prepared, feature1, feature2)) return 1;
    return 0;
}

int32_t idol_encode_dino(IdolEngine* engine, const float* prepared, float* feature1) {
    if (!engine || !prepared || !feature1) return fail("idol_encode_dino: null argument");
    CtxGuard guard(engine->ctx);
    return encode_dino(*engine, prepared, feature1);
}

int32_t idol_encode_vae(IdolEngine* engine, const float* prepared, float* feature2) {
    if (!engine || !prepared || !feature2) return fail("idol_encode_vae: null argument");
    CtxGuard guard(engine->ctx);
    return encode_vae(*engine, prepared, feature2);
}

int32_t idol_velocity(IdolEngine* engine, const float* feature1, const float* feature2,
                      const float* latent, const float* camera, float t, float* out_latent,
                      float* out_camera) {
    if (!engine || !feature1 || !feature2 || !latent || !camera || !out_latent || !out_camera)
        return fail("idol_velocity: null argument");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    Tensor f1 = upload_f16(E, 4101, 1280, feature1, 1280);
    Tensor f2 = upload_f16(E, 4101, 128, feature2, 128);
    Tensor lat = upload_f16(E, 8192, 16, latent, 16);
    Tensor cam = upload_f16(E, 1, 5, camera, 5);
    if (!f1.ok() || !f2.ok() || !lat.ok() || !cam.ok()) return 1;
    Tensor vlat, vcam;
    if (run_velocity(E, f1, f2, lat, cam, t, vlat, vcam)) return 1;
    int rc = download_f32(E, vlat, 8192, 16, out_latent);
    if (!rc) rc = download_f32(E, vcam, 1, 5, out_camera);
    E.rel(f1); E.rel(f2); E.rel(lat); E.rel(cam);
    return rc;
}

int32_t idol_denoise_from(IdolEngine* engine, const float* feature1, const float* feature2,
                          const float* latent0, const float* camera0, int steps, float cfg_strength,
                          float rescale_t, float* out_latent, float* out_camera,
                          IdolProgress progress, void* user) {
    if (!engine || !feature1 || !feature2 || !latent0 || !camera0 || !out_latent || !out_camera)
        return fail("idol_denoise: null argument");
    if (steps < 1 || steps > 1000) return fail("steps out of range");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    Tensor f1 = upload_f16(E, 4101, 1280, feature1, 1280);
    Tensor f2 = upload_f16(E, 4101, 128, feature2, 128);
    Tensor z1 = upload_f16(E, 4101, 1280, nullptr, 1280);
    Tensor z2 = upload_f16(E, 4101, 128, nullptr, 128);
    Tensor lat = upload_f16(E, 8192, 16, latent0, 16);
    Tensor cam = upload_f16(E, 1, 5, camera0, 5);
    if (!f1.ok() || !f2.ok() || !z1.ok() || !z2.ok() || !lat.ok() || !cam.ok()) return 1;
    double camf[5];
    for (int i = 0; i < 5; ++i) camf[i] = camera0[i];

    std::vector<double> tseq((size_t)steps + 1);
    for (int i = 0; i <= steps; ++i) {
        double v = 1.0 - (double)i / (double)steps;
        tseq[i] = (double)rescale_t * v / (1.0 + ((double)rescale_t - 1.0) * v);
    }
    float cam1[5];
    for (int step = 0; step < steps; ++step) {
        float t = (float)(1000.0 * tseq[step]);
        double dt = tseq[step] - tseq[step + 1];
        Tensor vp, cp, vn, cn;
        if (denoise_forward(E, f1, f2, lat, cam, t, vp, cp)) return 1;
        if (denoise_forward(E, z1, z2, lat, cam, t, vn, cn)) return 1;
        int n = 8192 * 16;
        float w = cfg_strength, dtf = (float)dt;
        int stride = lat.c;
        void* a[] = {&lat.p, &vp.p, &vn.p, &n, &stride, &w, &dtf};
        if (E.launch("k_cfg_update_strided", dim3((n + 255) / 256), dim3(256), 0, a)) return 1;
        if (download_f32(E, cp, 1, 5, cam1)) return 1;
        float cam2[5];
        if (download_f32(E, cn, 1, 5, cam2)) return 1;
        float cold[5];
        if (download_f32(E, cam, 1, 5, cold)) return 1;
        for (int i = 0; i < 5; ++i) {
            double v = (1.0 + cfg_strength) * cam1[i] - cfg_strength * cam2[i];
            camf[i] = (double)cold[i] - v * dt;
        }
        float cam_new[5];
        for (int i = 0; i < 5; ++i) cam_new[i] = (float)camf[i];
        E.zero(cam, (int64_t)cam.r * cam.c);
        __half ch[5];
        for (int i = 0; i < 5; ++i) ch[i] = __float2half(cam_new[i]);
        E.drv.htod(cam.p, ch, sizeof(ch));
        E.rel(vp); E.rel(vn); E.rel(cp); E.rel(cn);
        if (progress) progress((step + 1) * 100 / steps, "sampling", user);
    }
    int rc = download_f32(E, lat, 8192, 16, out_latent);
    if (!rc) {
        for (int i = 0; i < 5; ++i) out_camera[i] = (float)camf[i];
    }
    E.rel(f1); E.rel(f2); E.rel(z1); E.rel(z2); E.rel(lat); E.rel(cam);
    // The sampling stage is done with the denoiser; hand the ~730 MiB back before the decoder runs.
    E.free_weights();
    E.pool.clear();
    return rc;
}

int32_t idol_denoise(IdolEngine* engine, const float* feature1, const float* feature2, uint64_t seed,
                     int steps, float cfg_strength, float rescale_t, float* out_latent,
                     float* out_camera, IdolProgress progress, void* user) {
    std::vector<float> lat((size_t)8192 * 16), cam(5);
    uint64_t state = seed * 0x9E3779B97F4A7C15ull + 0x123456789ABCDEFull;
    fill_normal(lat.data(), lat.size(), state);
    fill_normal(cam.data(), cam.size(), state);
    return idol_denoise_from(engine, feature1, feature2, lat.data(), cam.data(), steps, cfg_strength,
                             rescale_t, out_latent, out_camera, progress, user);
}

int32_t idol_octree(IdolEngine* engine, const float* latent, int num_points, int level, uint64_t seed,
                    float* anchors) {
    if (!engine || !latent || !anchors) return fail("idol_octree: null argument");
    if (num_points < 1 || level < 1 || level > 10) return fail("idol_octree: bad arguments");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    Tensor lat = upload_f16(E, 8192, 16, latent, 16);
    if (!lat.ok()) return 1;
    std::vector<float> out;
    if (run_octree(E, lat, num_points, level, seed, out)) return 1;
    memcpy(anchors, out.data(), (size_t)num_points * 3 * sizeof(float));
    E.rel(lat);
    E.free_weights();
    E.pool.clear();
    return 0;
}

int32_t idol_gaussians(IdolEngine* engine, const float* latent, const float* anchors, int count,
                       float* xyz, float* color, float* scaling, float* rotation, float* opacity) {
    if (!engine || !latent || !anchors || !xyz || !color || !scaling || !rotation || !opacity)
        return fail("idol_gaussians: null argument");
    if (count < 1) return fail("idol_gaussians: bad count");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    const int per = 32;
    Tensor lat = upload_f16(E, 8192, 16, latent, 16);
    Tensor anc = upload_f16(E, count, 3, anchors, 3);
    if (!lat.ok() || !anc.ok()) return 1;
    CUdeviceptr posf = E.pool.alloc((size_t)count * 3 * 4);
    if (!posf) return fail("oom anchors");
    E.drv.htod(posf, anchors, (size_t)count * 3 * 4);
    const int FN = 170;
    float freqs[FN];
    for (int i = 0; i < FN; ++i) freqs[i] = (float)std::pow(2.0, 10.0 * i / (FN - 1.0));
    CUdeviceptr fq = E.pool.alloc(FN * 4);
    E.drv.htod(fq, freqs, FN * 4);
    Tensor feats = gs_forward(E, anc, posf, count, lat, fq);
    if (!feats.ok()) return 1;
    const int out_cols = 480;
    std::vector<float> raw((size_t)count * out_cols);
    if (download_f32(E, feats, count, out_cols, raw.data())) return 1;

    const Weights::Entry* pert = E.weights.find("decoder_gs.points_offset_perturbation");
    const Weights::Entry* base = E.weights.find("decoder_gs.base_offset_scale");
    if (!pert || !base) return fail("missing decoder_gs offset tables");
    const int pstride = pert->dims[1];
    auto pert_at = [&](int g, int k) {
        size_t idx = (size_t)g * pstride + k;
        if (pert->dtype == 0) return h2f(((const uint16_t*)pert->data)[idx]);
        return ((const float*)pert->data)[idx];
    };
    float base_scale = base->dtype == 0 ? h2f(((const uint16_t*)base->data)[0]) : ((const float*)base->data)[0];

    const float S = 0.5f * 1.5f;
    const float inv_sp = inv_softplus(0.004f);
    const float inv_sig = std::log(0.1f / 0.9f);
    const float kernel2 = 0.0009f * 0.0009f;
    for (int a = 0; a < count; ++a) {
        const float* f = raw.data() + (size_t)a * out_cols;
        for (int g = 0; g < per; ++g) {
            size_t gi = (size_t)a * per + g;
            float ox = 0, oy = 0, oz = 0;
            float sp = softplusf(f[448 + g] + base_scale) * S;
            auto off = [&](int k) {
                float v = f[g * 3 + k] + pert_at(g, k);
                return std::tanh(v) * sp;
            };
            ox = off(0); oy = off(1); oz = off(2);
            xyz[gi * 3 + 0] = (anchors[a * 3 + 0] + ox) - 0.5f;
            xyz[gi * 3 + 1] = (anchors[a * 3 + 1] + oy) - 0.5f;
            xyz[gi * 3 + 2] = (anchors[a * 3 + 2] + oz) - 0.5f;
            for (int k = 0; k < 3; ++k) {
                color[gi * 3 + k] = f[96 + g * 3 + k];
                float s = softplusf(f[192 + g * 3 + k] + inv_sp);
                scaling[gi * 3 + k] = std::sqrt(s * s + kernel2);
            }
            float q[4] = {f[288 + g * 4 + 0] * 0.1f + 1.0f, f[288 + g * 4 + 1] * 0.1f,
                          f[288 + g * 4 + 2] * 0.1f, f[288 + g * 4 + 3] * 0.1f};
            float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (norm < 1e-12f) norm = 1e-12f;
            for (int k = 0; k < 4; ++k) rotation[gi * 4 + k] = q[k] / norm;
            opacity[gi] = sigmoidf(f[416 + g] + inv_sig);
        }
    }
    E.rel(feats);
    E.free_weights();
    E.pool.clear();
    return 0;
}
namespace {

bool utf8_to_wide(const char* s, std::wstring& out) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return false;
    out.assign((size_t)n, L'\0');
    return MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), n) == n;
}

int write_ply(const char* path, int count, const float* xyz, const float* color, const float* scaling,
              const float* rotation, const float* opacity) {
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) return fail("bad output path");
    FILE* f = _wfopen(wide.c_str(), L"wb");
    if (!f) return fail("cannot create %s", path);
    const char* header =
        "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
        "property float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n";
    fprintf(f, header, count);
    std::vector<float> row(17);
    const float qw = 0.7071067811865476f, qx = 0.7071067811865476f;
    const float eps = 1e-6f;
    double bytes = 0;
    for (int i = 0; i < count; ++i) {
        const float* p = xyz + (size_t)i * 3;
        row[0] = p[0];
        row[1] = -p[2];
        row[2] = p[1];
        row[3] = row[4] = row[5] = 0.f;
        row[6] = color[i * 3 + 0];
        row[7] = color[i * 3 + 1];
        row[8] = color[i * 3 + 2];
        float op = opacity[i];
        op = op < eps ? eps : (op > 1.f - eps ? 1.f - eps : op);
        row[9] = std::log(op / (1.f - op));
        for (int k = 0; k < 3; ++k) {
            float s = scaling[i * 3 + k];
            row[10 + k] = std::log(s > eps ? s : eps);
        }
        float aw = qw, ax = qx, ay = 0.f, az = 0.f;
        float bw = rotation[i * 4 + 0], bx = rotation[i * 4 + 1];
        float by = rotation[i * 4 + 2], bz = rotation[i * 4 + 3];
        row[13] = aw * bw - ax * bx - ay * by - az * bz;
        row[14] = aw * bx + ax * bw + ay * bz - az * by;
        row[15] = aw * by - ax * bz + ay * bw + az * bx;
        row[16] = aw * bz + ax * by - ay * bx + az * bw;
        fwrite(row.data(), sizeof(float), row.size(), f);
        bytes += 68;
    }
    fclose(f);
    return 0;
}

int write_camera_json(const char* path, const float* cam) {
    float dx = cam[0], dy = cam[1], dz = cam[2];
    float norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(norm > 1e-6f)) { dx = 0.f; dy = 0.f; dz = 1.f; norm = 1.f; }
    dx /= norm; dy /= norm; dz /= norm;
    float invd = cam[3], halfw = cam[4];
    double radius, fov;
    if (invd < 1e-3f) {
        radius = 1000.0;
        fov = 2.0 * std::atan((double)halfw / radius) * 57.29577951308232;
    } else {
        radius = 1.0 / (double)invd;
        fov = 2.0 * std::atan((double)halfw * (double)invd) * 57.29577951308232;
    }
    double yaw = std::atan2((double)dx, (double)dy);
    double pitch = std::asin((double)(dz > 1.f ? 1.f : (dz < -1.f ? -1.f : dz)));
    std::wstring wide;
    if (!utf8_to_wide(path, wide)) return fail("bad camera path");
    FILE* f = _wfopen(wide.c_str(), L"wb");
    if (!f) return fail("cannot create %s", path);
    fprintf(f, "{\n  \"yaw\": %.9g,\n  \"pitch\": %.9g,\n  \"radius\": %.9g,\n  \"fov\": %.9g\n}\n",
            yaw, pitch, radius, fov);
    fclose(f);
    return 0;
}

int alpha_heuristic(const float* rgba, int w, int h, float* alpha) {
    double br = 0, bg = 0, bb = 0;
    int count = 0;
    for (int side = 0; side < 4; ++side) {
        for (int t = 0; t < 5; ++t) {
            int x = side == 0 ? t : (side == 1 ? w - 1 - t : (side == 2 ? t * (w - 1) / 4 : w - 1 - t * (w - 1) / 4));
            int y = side == 2 || side == 3 ? t : (side == 0 ? t * (h - 1) / 4 : h - 1 - t * (h - 1) / 4);
            if (x < 0) x = 0;
            if (x >= w) x = w - 1;
            if (y < 0) y = 0;
            if (y >= h) y = h - 1;
            const float* p = rgba + ((size_t)y * w + x) * 4;
            br += p[0]; bg += p[1]; bb += p[2];
            ++count;
        }
    }
    br /= count; bg /= count; bb /= count;
    for (int i = 0; i < w * h; ++i) {
        float d = std::fabs(rgba[i * 4 + 0] - (float)br);
        d = std::fmax(d, std::fabs(rgba[i * 4 + 1] - (float)bg));
        d = std::fmax(d, std::fabs(rgba[i * 4 + 2] - (float)bb));
        float v = (d - 0.06f) / 0.06f;
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        alpha[i] = v * v * (3.f - 2.f * v);
    }
    return 0;
}

#include "birefnet.inc"

}  // namespace


int32_t idol_preprocess_mode(IdolEngine* engine, const float* rgba, int width, int height,
                             float crop_margin, float* prepared, float* alpha) {
    if (!rgba || !prepared || width < 8 || height < 8) return fail("idol_preprocess: bad arguments");
    if (!(crop_margin > 1.0f) || crop_margin > 4.0f) crop_margin = 1.2f;
    const int S = 1024;
    double scale = (double)S / (double)(width < height ? width : height);
    int rw = (int)std::lround((double)width * scale);
    int rh = (int)std::lround((double)height * scale);
    if (rw < 8) rw = 8;
    if (rh < 8) rh = 8;
    std::vector<float> resized((size_t)rw * rh * 4);
    resize_planar(rgba, width, height, 4, resized.data(), rw, rh, 4);

    bool has_alpha = false;
    for (size_t i = 0; i < (size_t)rw * rh; ++i) {
        if (resized[i * 4 + 3] < 0.996f) { has_alpha = true; break; }
    }
    std::vector<float> mask((size_t)rw * rh, 1.0f);
    const bool matte = engine == nullptr || engine->remove_background;
    if (has_alpha) {
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = resized[i * 4 + 3];
    } else if (engine && engine->remove_background) {
        if (birefnet_mask(*engine, resized.data(), rw, rh, mask.data())) return 1;
    } else if (!engine) {
        alpha_heuristic(resized.data(), rw, rh, mask.data());
    }

    std::vector<float> eroded((size_t)rw * rh);
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            float m = 1.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= rh || xx < 0 || xx >= rw) continue;
                    m = std::fmin(m, mask[(size_t)yy * rw + xx]);
                }
            }
            eroded[(size_t)y * rw + x] = m;
        }
    }
    int left = rw, top = rh, right = -1, bottom = -1;
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            if (eroded[(size_t)y * rw + x] > 0.f) {
                if (x < left) left = x;
                if (x > right) right = x;
                if (y < top) top = y;
                if (y > bottom) bottom = y;
            }
        }
    }
    if (right < 0) {
        left = 0; top = 0; right = rw - 1; bottom = rh - 1;
    }
    double cx = (left + right) * 0.5, cy = (top + bottom) * 0.5;
    double halfspan =
        (double)(right - left > bottom - top ? right - left : bottom - top) * 0.5 * (double)crop_margin;
    // Without matting there is no subject bounding box: keep the largest centred square instead of
    // growing the crop past the frame edges.
    if (!matte) halfspan = 0.5 * (double)(rw < rh ? rw : rh);
    if (halfspan < 8.0) halfspan = 8.0;
    int x0 = (int)(cx - halfspan), y0 = (int)(cy - halfspan);
    int cw = (int)(cx + halfspan) - x0;
    int ch = (int)(cy + halfspan) - y0;
    if (cw < 8) cw = 8;
    if (ch < 8) ch = 8;

    std::vector<float> crop((size_t)cw * ch * 4, 0.f);
    for (int y = 0; y < ch; ++y) {
        int sy = y0 + y;
        if (sy < 0 || sy >= rh) continue;
        for (int x = 0; x < cw; ++x) {
            int sx = x0 + x;
            if (sx < 0 || sx >= rw) continue;
            size_t di = ((size_t)y * cw + x) * 4;
            size_t si = ((size_t)sy * rw + sx) * 4;
            crop[di + 0] = resized[si + 0];
            crop[di + 1] = resized[si + 1];
            crop[di + 2] = resized[si + 2];
            crop[di + 3] = resized[si + 3];
        }
    }
    std::vector<float> out((size_t)S * S * 4);
    resize_planar(crop.data(), cw, ch, 4, out.data(), S, S, 4);
    for (int i = 0; i < S * S; ++i) {
        float a = out[i * 4 + 3];
        a = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
        prepared[(size_t)0 * S * S + i] = out[i * 4 + 0] * a;
        prepared[(size_t)1 * S * S + i] = out[i * 4 + 1] * a;
        prepared[(size_t)2 * S * S + i] = out[i * 4 + 2] * a;
    }
    if (alpha) {
        std::vector<float> a1((size_t)rw * rh);
        for (size_t i = 0; i < a1.size(); ++i) a1[i] = eroded[i];
        std::vector<float> a2((size_t)width * height);
        resize_planar(a1.data(), rw, rh, 1, a2.data(), width, height, 1);
        memcpy(alpha, a2.data(), a2.size() * sizeof(float));
    }
    return 0;
}

int32_t idol_preprocess(IdolEngine* engine, const float* rgba, int width, int height, float* prepared,
                        float* alpha) {
    return idol_preprocess_mode(engine, rgba, width, height, 1.2f, prepared, alpha);
}

int32_t idol_generate(IdolEngine* engine, const float* rgba, int width, int height, uint64_t seed,
                      int steps, int num_gaussians, const char* out_ply, const char* out_camera_json,
                      IdolProgress progress, void* user) {
    if (!engine || !rgba) return fail("idol_generate: null argument");
    if (num_gaussians < 32) return fail("num_gaussians too small");
    const int per = 32;
    int anchors_n = num_gaussians / per;
    std::vector<float> prepared((size_t)3 * 1024 * 1024);
    if (progress) progress(1, "preprocess", user);
    if (idol_preprocess_mode(engine, rgba, width, height, engine->crop_margin, prepared.data(), nullptr))
        return 1;
    std::vector<float> feature1((size_t)4101 * 1280), feature2((size_t)4101 * 128);
    if (progress) progress(4, "encode", user);
    if (idol_encode(engine, prepared.data(), feature1.data(), feature2.data())) return 1;
    std::vector<float> latent((size_t)8192 * 16), camera(5);
    if (progress) progress(12, "denoise", user);
    StepProgress sampling{progress, user, 12, 55};
    if (idol_denoise(engine, feature1.data(), feature2.data(), seed, steps, 3.0f, 3.0f,
                     latent.data(), camera.data(), step_progress, &sampling)) return 1;
    std::vector<float> anchors((size_t)anchors_n * 3);
    if (progress) progress(55, "octree", user);
    if (idol_octree(engine, latent.data(), anchors_n, 8, seed, anchors.data())) return 1;
    std::vector<float> xyz((size_t)anchors_n * per * 3), color((size_t)anchors_n * per * 3);
    std::vector<float> scaling((size_t)anchors_n * per * 3), rotation((size_t)anchors_n * per * 4);
    std::vector<float> opacity((size_t)anchors_n * per);
    if (progress) progress(65, "gaussians", user);
    if (idol_gaussians(engine, latent.data(), anchors.data(), anchors_n, xyz.data(), color.data(),
                       scaling.data(), rotation.data(), opacity.data())) return 1;
    if (progress) progress(92, "export", user);
    if (out_ply && write_ply(out_ply, anchors_n * per, xyz.data(), color.data(), scaling.data(),
                             rotation.data(), opacity.data())) return 1;
    if (out_camera_json && write_camera_json(out_camera_json, camera.data())) return 1;
    if (g_profile) prof_dump("generate");
    if (progress) progress(100, "done", user);
    return 0;
}

int32_t idol_debug_vae(IdolEngine* engine, const float* prepared, float* out) {
    if (!engine || !prepared || !out) return fail("idol_debug_vae: null argument");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    Tensor x = vae_encode(E, prepared);
    if (!x.ok()) return 1;
    int rc = download_f32(E, x, 128 * 128, 64, out);
    E.rel(x);
    return rc;
}

int32_t idol_debug_gemm(IdolEngine* engine, const float* a, int m, int k, const float* b, int n,
                        float* c) {
    if (!engine || !a || !b || !c) return fail("idol_debug_gemm: null argument");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    int kp = (k + 15) / 16 * 16, np = (n + 63) / 64 * 64;
    std::vector<__half> ha((size_t)m * kp, __half(0.f)), hb((size_t)k * n, __half(0.f));
    for (int r = 0; r < m; ++r)
        for (int cc = 0; cc < k; ++cc) ha[(size_t)r * kp + cc] = __float2half(a[(size_t)r * k + cc]);
    for (int r = 0; r < k; ++r)
        for (int cc = 0; cc < n; ++cc) hb[(size_t)r * n + cc] = __float2half(b[(size_t)r * n + cc]);
    CUdeviceptr da = 0, db = 0, dc = 0;
    cuMemAlloc(&da, ha.size() * 2);
    cuMemAlloc(&db, hb.size() * 2);
    cuMemAlloc(&dc, (size_t)m * np * 2);
    E.drv.htod(da, ha.data(), ha.size() * 2);
    E.drv.htod(db, hb.data(), hb.size() * 2);
    int rc = E.gemm_raw(da, kp, m, kp, db, n, 0, dc, np, 0);
    std::vector<__half> hc((size_t)m * np);
    if (!rc) {
        E.drv.dtoh(hc.data(), dc, hc.size() * 2);
        if (E.sync()) rc = 1;
    }
    if (!rc) {
        for (int r = 0; r < m; ++r)
            for (int cc = 0; cc < n; ++cc) c[(size_t)r * n + cc] = __half2float(hc[(size_t)r * np + cc]);
    }
    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    return rc;
}

int32_t idol_debug_flash(IdolEngine* engine, const float* q, const float* kk, const float* v, int l,
                         int heads, float* out) {
    if (!engine || !q || !kk || !v || !out) return fail("idol_debug_flash: null argument");
    CtxGuard guard(engine->ctx);
    IdolEngine& E = *engine;
    size_t n = (size_t)l * heads * 64;
    CUdeviceptr dq = 0, dk = 0, dv = 0, do_ = 0;
    cuMemAlloc(&dq, n * 2); cuMemAlloc(&dk, n * 2); cuMemAlloc(&dv, n * 2); cuMemAlloc(&do_, n * 2);
    std::vector<__half> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2half(q[i]);
    E.drv.htod(dq, tmp.data(), n * 2);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2half(kk[i]);
    E.drv.htod(dk, tmp.data(), n * 2);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2half(v[i]);
    E.drv.htod(dv, tmp.data(), n * 2);
    int stride = heads * 64;
    float scale = 1.0f / std::sqrt(64.0f);
    void* args[] = {&dq, &dk, &dv, &do_, &l, &l, &heads, &stride, &stride, &scale};
    int rc = E.launch("k_flash3", dim3((l + 63) / 64, heads), dim3(128), kFlashSmem, args);
    if (!rc) {
        E.drv.dtoh(tmp.data(), do_, n * 2);
        if (E.sync()) rc = 1;
    }
    if (!rc) for (size_t i = 0; i < n; ++i) out[i] = __half2float(tmp[i]);
    cuMemFree(dq); cuMemFree(dk); cuMemFree(dv); cuMemFree(do_);
    return rc;
}

}  // extern "C"
