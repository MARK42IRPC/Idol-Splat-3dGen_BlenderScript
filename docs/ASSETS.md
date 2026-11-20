# Model assets

The repository holds code only. A working install needs two files inside
`is3d/native/`:

| File | Size | Purpose |
| --- | --- | --- |
| `idol_engine.dll` | ~1.2 MB | the inference engine (host code + embedded PTX) |
| `idol_weights.bin` | ~3.54 GB | every tensor the pipeline uses, one flat file |

`idol_weights.bin` layout (`IDOLW1` header):

```
u8  magic[8] = "IDOLW1\0\0"
u32 tensor_count
u32 reserved
u64 data_offset
tensor_count * {
    u32 name_len, u8 name[name_len],
    i32 rank, i32 dims[rank],
    i32 dtype,          // 0 = fp16, 1 = fp32
    u64 offset, u64 elements
}
```

Weights are memory-mapped from the file and uploaded to the GPU stage by stage,
so the file can live on a slow disk without hurting steady-state speed.

## Upstream sources

The release is a repack, not the original distribution. Everything in
`idol_weights.bin` comes from these projects; the packer keeps only what the
inference path touches and re-lays-out the weights for the kernels.

| Upstream | Licence-holder | What we take |
| --- | --- | --- |
| [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk) | speridlabs | denoiser, VAE decoder, Gaussian decoder |
| [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) | Black Forest Labs | FLUX.2 klein 4B VAE encoder |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Meta | DINOv3 ViT-H/16+ |
| [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | Peng Zheng et al. | matte network (Swin-L + deformable ASPP) |

Hugging Face mirrors of the same upstreams: `huggingface.co/speridlabs`,
`huggingface.co/black-forest-labs`, `huggingface.co/facebook`,
`huggingface.co/ZhengPeng7`.

## Where the installer looks

`install/install.ps1` resolves one base URL and then fetches, relative to it:

```
is3d-addon-<version>.zip
SHA256SUMS.txt
idol_engine.dll
idol_weights.bin.part00 .. partNN
```

`SHA256SUMS.txt` is `<sha256>  <name>` per line. The weight slices are
concatenated in name order and the result is verified again against the
`idol_weights.bin` entry, so a truncated or reordered mirror cannot produce a
half-working install. Parts download with `curl -C -`, so an interrupted install
resumes where it stopped; re-running with `-Force` starts clean.

Any mirror only has to expose those names at one flat level to be usable.

## Rebuilding the weight file

See `docs/BUILD.md`. The packer keeps a whitelist of tensors, so the file is
smaller than the sum of the upstream checkpoints and no training-only buffers
are shipped.
