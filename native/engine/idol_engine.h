#ifndef IDOL_ENGINE_H
#define IDOL_ENGINE_H

#include <stdint.h>

#ifdef IDOL_ENGINE_BUILD
#define IDOL_API __declspec(dllexport)
#else
#define IDOL_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IdolEngine IdolEngine;
typedef void (*IdolProgress)(int percent, const char* message, void* user);

IDOL_API uint32_t idol_engine_abi(void);
IDOL_API const char* idol_engine_last_error(void);
IDOL_API int32_t idol_engine_create(int32_t device, const char* weights_path, IdolEngine** out);
IDOL_API void idol_engine_destroy(IdolEngine* engine);
IDOL_API int32_t idol_engine_info(IdolEngine* engine, char* name, int name_len, int32_t* major,
                                  int32_t* minor, uint64_t* free_bytes, uint64_t* total_bytes);
IDOL_API int32_t idol_memory(IdolEngine* engine, uint64_t* used, uint64_t* peak);
IDOL_API int32_t idol_engine_reset_peak(IdolEngine* engine);
IDOL_API int32_t idol_engine_set_crop_margin(IdolEngine* engine, float margin);
IDOL_API int32_t idol_engine_set_remove_background(IdolEngine* engine, int32_t enabled);
IDOL_API int32_t idol_engine_unload(IdolEngine* engine);

// RGBA float32 (top-down, sRGB 0..1, 4 channels) -> prepared planar RGB 3*1024*1024 float32,
// optional alpha output width*height (uint8 stored as float 0..1).
IDOL_API int32_t idol_preprocess(IdolEngine* engine, const float* rgba, int width, int height,
                                 float* prepared, float* alpha);
IDOL_API int32_t idol_preprocess_mode(IdolEngine* engine, const float* rgba, int width, int height,
                                      float crop_margin, float* prepared, float* alpha);
IDOL_API int32_t idol_encode(IdolEngine* engine, const float* prepared, float* feature1, float* feature2);
IDOL_API int32_t idol_encode_dino(IdolEngine* engine, const float* prepared, float* feature1);
IDOL_API int32_t idol_encode_vae(IdolEngine* engine, const float* prepared, float* feature2);
IDOL_API int32_t idol_velocity(IdolEngine* engine, const float* feature1, const float* feature2,
                               const float* latent, const float* camera, float t, float* out_latent,
                               float* out_camera);
IDOL_API int32_t idol_denoise(IdolEngine* engine, const float* feature1, const float* feature2,
                              uint64_t seed, int steps, float cfg_strength, float rescale_t,
                              float* out_latent, float* out_camera, IdolProgress progress, void* user);
IDOL_API int32_t idol_denoise_from(IdolEngine* engine, const float* feature1, const float* feature2,
                                   const float* latent0, const float* camera0, int steps,
                                   float cfg_strength, float rescale_t, float* out_latent,
                                   float* out_camera, IdolProgress progress, void* user);
IDOL_API int32_t idol_octree(IdolEngine* engine, const float* latent, int num_points, int level,
                             uint64_t seed, float* anchors);
IDOL_API int32_t idol_gaussians(IdolEngine* engine, const float* latent, const float* anchors, int count,
                                float* xyz, float* color, float* scaling, float* rotation, float* opacity);
IDOL_API int32_t idol_generate(IdolEngine* engine, const float* rgba, int width, int height, uint64_t seed,
                               int steps, int num_gaussians, const char* out_ply,
                               const char* out_camera_json, IdolProgress progress, void* user);

// Debug hooks used by the validation harness.
IDOL_API int32_t idol_debug_gemm(IdolEngine* engine, const float* a, int m, int k, const float* b, int n,
                                 float* c);
IDOL_API int32_t idol_debug_flash(IdolEngine* engine, const float* q, const float* kk, const float* v, int l,
                                  int heads, float* out);
// VAE encoder posterior (pre-sampling): out is [128*128][64] = mean[32] | logvar[32].
IDOL_API int32_t idol_debug_vae(IdolEngine* engine, const float* prepared, float* out);

#ifdef __cplusplus
}
#endif
#endif
