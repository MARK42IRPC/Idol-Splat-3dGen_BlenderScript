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

## Where the installer looks

1. `-AssetBaseUrl` if given,
2. otherwise `https://github.com/<repo>/releases/download/v<version>/`.

The expected files are `idol_engine.dll`, `idol_weights.bin.part00 .. partNN` and
`SHA256SUMS.txt` (`<sha256>  <name>` per line). Mirrors only need to expose the
same three names; point `-AssetBaseUrl` at them, for example a Hugging Face or
ModelScope repository.

## Rebuilding the weight file

See `docs/BUILD.md`. The packer keeps a whitelist of tensors, so the file is
smaller than the sum of the upstream checkpoints and no training-only buffers
are shipped.
