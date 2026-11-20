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

## Mirrors

All three carry byte-identical files and the same `SHA256SUMS.txt`:

| `-Mirror` | Base URL |
| --- | --- |
| `github` (default) | `https://github.com/MARK42IRPC/is3D-BlenderScript/releases/download/v<version>` |
| `hf` | `https://hf-mirror.com/Mark42IRP/is3D-BlenderScript/resolve/main` |
| `modelscope` | `https://www.modelscope.cn/models/Mark42IRPC/is3D-BlenderScript/resolve/master` |

`-AssetBaseUrl <url>` overrides all of them (an internal file server, MinIO, a
`python -m http.server` on a LAN share, ...).

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
