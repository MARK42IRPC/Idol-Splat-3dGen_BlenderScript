"""Run the torch BiRefNet reference on the same input the engine sees and dump intermediates."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".reference" / "idol-sdk"))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from idol.models.birefnet import BiRefNet

OUT = ROOT / "validation" / "results" / "birefnet-golden" / "ref"
GOLDEN = ROOT / "validation" / "results" / "birefnet-golden" / "snowman.npz"


def i2p(image, ref):
    b, c, hf, wf = image.shape
    hg, wg = hf // ref.shape[-2], wf // ref.shape[-1]
    h, w = hf // hg, wf // wg
    return image.view(b, c, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c * hg * wg, h, w)


def feed():
    golden = np.load(GOLDEN)
    inv_std = np.array([0.229, 0.224, 0.225], np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    rgb = golden["input"][0].astype(np.float32) * inv_std[:, None, None] + mean[:, None, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    t = torch.from_numpy(rgb)[None].cuda()
    norm = torch.tensor(mean).view(1, 3, 1, 1).cuda()
    std = torch.tensor(inv_std).view(1, 3, 1, 1).cuda()
    return ((t - norm) / std).half()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = BiRefNet().cuda().half()
    sd = load_file(str(ROOT / "models" / "speridlabs" / "idol" / "birefnet.safetensors"))
    missing, unused = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), "unused", len(unused), flush=True)
    model.eval()

    wanted = ["bb.patch_embed", "bb.layers.0.blocks.0", "bb.layers.0.blocks.1", "bb.layers.0.blocks.1.norm1", "bb.layers.0.blocks.1.attn", "bb.layers.0.blocks.1.mlp", "bb.layers.0.blocks.1.norm2", "bb.layers.0.downsample"]
    store = {}

    def hook(name):
        def run(_m, _i, out):
            if isinstance(out, torch.Tensor):
                store.setdefault(name, out.detach().float())
        return run

    handles = []
    for name, module in model.named_modules():
        if name in wanted:
                    handles.append(module.register_forward_hook(hook("bbk_" + name.replace(".", "_"))))

    x = feed()
    with torch.no_grad():
        alpha = model(x)
    for h in handles:
        h.remove()
    out = {k: v.cpu().numpy() for k, v in store.items()}
    out["alpha_full"] = alpha.float().cpu().numpy()

    with torch.no_grad():
        x1a, x2a, x3a, x4a = model.bb(x)
        B, C, H, W = x.shape
        x1b, x2b, x3b, x4b = model.bb(F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True))
        u = lambda t, r: F.interpolate(t, size=r.shape[2:], mode="bilinear", align_corners=True)
        x1 = torch.cat([x1a, u(x1b, x1a)], 1)
        x2 = torch.cat([x2a, u(x2b, x2a)], 1)
        x3 = torch.cat([x3a, u(x3b, x3a)], 1)
        x4 = torch.cat([x4a, u(x4b, x4a)], 1)
        x4c = torch.cat([u(x1, x4), u(x2, x4), u(x3, x4), x4], 1)
        out["x4c"] = x4c.float().cpu().numpy()
        squeeze = model.squeeze_module(x4c)
        out["squeeze"] = squeeze.float().cpu().numpy()
        dec = model.decoder
        ipt5 = dec.ipt_blk5(i2p(x, x4c))
        out["ipt5"] = ipt5.float().cpu().numpy()
        p4 = dec.decoder_block4(torch.cat((squeeze, ipt5), 1))
        out["p4raw"] = p4.float().cpu().numpy()
        p4 = p4 * dec.gdt_convs_attn_4(dec.gdt_convs_4(p4)).sigmoid()
        out["p4"] = p4.float().cpu().numpy()
        _p3 = u(p4, x3) + dec.lateral_block4(x3)
        ipt4 = dec.ipt_blk4(i2p(x, _p3))
        out["ipt4"] = ipt4.float().cpu().numpy()
        out["_p3"] = _p3.float().cpu().numpy()
        p3 = dec.decoder_block3(torch.cat((_p3, ipt4), 1))
        out["p3raw"] = p3.float().cpu().numpy()
        p3 = p3 * dec.gdt_convs_attn_3(dec.gdt_convs_3(p3)).sigmoid()
        out["p3"] = p3.float().cpu().numpy()
        _p2 = u(p3, x2) + dec.lateral_block3(x2)
        ipt3 = dec.ipt_blk3(i2p(x, _p2))
        out["ipt3"] = ipt3.float().cpu().numpy()
        out["_p2"] = _p2.float().cpu().numpy()
        p2 = dec.decoder_block2(torch.cat((_p2, ipt3), 1))
        out["p2raw"] = p2.float().cpu().numpy()
        p2 = p2 * dec.gdt_convs_attn_2(dec.gdt_convs_2(p2)).sigmoid()
        out["p2"] = p2.float().cpu().numpy()
        _p1 = u(p2, x1) + dec.lateral_block2(x1)
        ipt2 = dec.ipt_blk2(i2p(x, _p1))
        out["ipt2"] = ipt2.float().cpu().numpy()
        d1 = dec.decoder_block1(torch.cat((_p1, ipt2), 1))
        out["dec1"] = d1.float().cpu().numpy()
        _p1 = u(d1, x)
        ipt1 = dec.ipt_blk1(i2p(x, _p1))
        out["ipt1"] = ipt1.float().cpu().numpy()
        logits = dec.conv_out1(torch.cat((_p1, ipt1), 1))
        out["logits"] = logits.float().cpu().numpy()
        out["alpha"] = torch.sigmoid(logits).float().cpu().numpy()

    for key, value in out.items():
        np.save(OUT / f"{key}.npy", value)
    print("saved", len(out), "tensors ->", OUT, flush=True)
    print("alpha coverage", float((out["alpha"][0, 0] > 0.5).mean()), flush=True)


if __name__ == "__main__":
    main()




