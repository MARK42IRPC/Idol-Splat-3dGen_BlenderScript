"""Pack the trimmed Idol weights into the native engine's single-file format."""
import json, struct, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
IDOL = MODELS / "speridlabs" / "idol"
DINO = MODELS / "LibreYOLO" / "LibreDINOv3vith16plus" / "model.safetensors"
VAE = MODELS / "black-forest-labs" / "FLUX.2-klein-4B" / "vae" / "diffusion_pytorch_model.safetensors"

SOURCES = [
    ("denoiser", IDOL / "denoiser.safetensors"),
    ("decoder", IDOL / "decoder.safetensors"),
    ("decoder_gs", IDOL / "decoder_gs.safetensors"),
    ("dinov3", DINO),
    ("vae", VAE),
    ("birefnet", IDOL / "birefnet.safetensors"),
]

def keep(prefix, key):
    if key.endswith("num_batches_tracked"):
        return False
    if prefix == "vae":
        return key.startswith("encoder.") or key.startswith("quant_conv.") or key.startswith("bn.")
    if prefix == "dinov3":
        return key != "embeddings.mask_token"
    if prefix == "birefnet":
        # mask_token is unused by the alpha head and relative_position_index is only a lookup table
        # we rebuild from the window geometry on the device.
        return key not in ("bb.patch_embed.mask_token",
                           "bb.layers.0.blocks.0.attn.relative_position_index") and not key.endswith(
            "attn.relative_position_index")
    if prefix == "denoiser":
        return key != "pos_pe"
    return True

def dtype_for(key):
    if ("running_mean" in key or "running_var" in key
            or key.endswith("freqs_0") or key.endswith("freqs_1") or key.endswith("freqs_2")
            or key.endswith("base_offset_scale")):
        return 1
    return 0


FORCE_DTYPE = {
    "dinov3.rope_cos": 1,
    "dinov3.rope_sin": 1,
}

def transform(prefix, key, t):
    if key.endswith(".weight") and t.dim() in (2, 4):
        if t.dim() == 2:
            t = t.t().contiguous()
        else:
            t = t.reshape(t.shape[0], -1).t().contiguous()
    elif t.dim() == 0:
        t = t.reshape(1)
    return t

def pad(t):
    if t.dim() == 1:
        n = t.shape[0]
        pad_n = (n + 63) // 64 * 64
        if pad_n != n:
            t = torch.cat([t, torch.zeros(pad_n - n, dtype=t.dtype)])
        return t
    if t.dim() == 2:
        r, c = t.shape
        rp, cp = (r + 15) // 16 * 16, (c + 63) // 64 * 64
        if (rp, cp) != (r, c):
            out = torch.zeros(rp, cp, dtype=t.dtype)
            out[:r, :c] = t
            t = out
        return t
    return t

def computed_tensors():
    out = {}
    # Denoiser positional embedding (1024 channels from 3D Sobol positions).
    with safe_open(str(IDOL / "denoiser.safetensors"), "pt") as f:
        pos = f.get_tensor("pos_pe")[0].float()
    channels, max_res = 1024, 16
    freq_dim = channels // 3 // 2
    freqs_2 = torch.arange(max_res, dtype=torch.float32)
    res_dim = max(0, freq_dim - max_res)
    freqs_res = torch.arange(res_dim, dtype=torch.float32) / res_dim * max_res
    freqs = 2 ** torch.cat([freqs_2, freqs_res])[:freq_dim]
    ang = torch.outer(pos.reshape(-1), freqs) * 2 * torch.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(-1, 3 * 2 * freq_dim)
    emb = torch.cat([emb, torch.zeros(emb.shape[0], channels - emb.shape[1])], dim=-1)
    out["denoiser.pos_embed"] = emb.half()

    # DINOv3 rotary embeddings for a 1024x1024 image (64x64 patches of 16 px).
    head_dim, base, n = 64, 100.0, 64
    inv = 1.0 / base ** torch.arange(0, 1, 4.0 / head_dim, dtype=torch.float32)
    ch = (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2 - 1
    coords = torch.stack(torch.meshgrid(ch, ch, indexing="ij"), dim=-1).reshape(-1, 2)
    ang = 2 * torch.pi * coords[:, :, None] * inv[None, None, :]
    ang = ang.flatten(1).tile(2)
    out["dinov3.rope_cos"] = torch.cos(ang).float().contiguous()
    out["dinov3.rope_sin"] = torch.sin(ang).float().contiguous()

    return out

def main():
    out_path = ROOT / "native" / "engine" / "build" / "idol_weights.bin"
    entries = []  # (name, dims, dtype, tensor_loader)
    for prefix, path in SOURCES:
        with safe_open(str(path), "pt") as f:
            for key in f.keys():
                if not keep(prefix, key):
                    continue
                name = f"{prefix}.{key}"
                sl = f.get_slice(key)
                shape = list(sl.get_shape())
                entries.append((name, prefix, key, shape))
    for name, t in computed_tensors().items():
        entries.append((name, None, None, list(t.shape)))
    print("tensors:", len(entries))

    # Pass 1: determine transformed shapes without materialising everything.
    final = []  # (name, dims, dtype, data)
    cache = {}
    for prefix, path in SOURCES:
        cache[prefix] = safe_open(str(path), "pt")
    computed = computed_tensors()
    total = 0
    for name, prefix, key, shape in entries:
        if prefix is None:
            t = computed[name]
        else:
            t = cache[prefix].get_tensor(key)
            if key.endswith("num_batches_tracked"):
                continue
            t = transform(prefix, key, t.float() if t.dtype == torch.float32 else t)
            if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
                t = t.half()
            else:
                t = t.float()
            t = pad(t)
        dt = FORCE_DTYPE.get(name, dtype_for(key or name))
        if dt == 0:
            t = t.half()
        elif dt == 2:
            t = t.int()
        else:
            t = t.float()
        t = t.contiguous().cpu()
        dims = list(t.shape)
        data = t.numpy().tobytes()
        final.append((name, dims, dt, data))
        total += len(data)
        del t, data
    print("data bytes:", total)

    header_entries = []
    off = 0
    for name, dims, dt, data in final:
        header_entries.append((name, dims, dt, off, len(data) // (2 if dt == 0 else 4) if dt != 2 else len(data)//4))
        off = (off + len(data) + 255) // 256 * 256
    data_off = 24 + sum(4 + len(n.encode()) + 4 + 4 * len(d) + 4 + 8 + 8 for n, d, _, _, _ in header_entries)
    data_off = (data_off + 255) // 256 * 256

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(b"IDOLW1\0\0")
        fh.write(struct.pack("<II", len(header_entries), 0))
        fh.write(struct.pack("<Q", data_off))
        for name, dims, dt, o, nelem in header_entries:
            nb = name.encode()
            fh.write(struct.pack("<I", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<i", len(dims)))
            for d in dims:
                fh.write(struct.pack("<i", d))
            fh.write(struct.pack("<i", dt))
            fh.write(struct.pack("<QQ", o, nelem))
        fh.write(b"\0" * (data_off - fh.tell()))
        for (name, dims, dt, data), (_, _, _, o, _) in zip(final, header_entries):
            fh.write(data)
            pad_to = ((len(data) + 255) // 256 * 256) - len(data)
            fh.write(b"\0" * pad_to)
    print("wrote", out_path, out_path.stat().st_size)

if __name__ == "__main__":
    main()

