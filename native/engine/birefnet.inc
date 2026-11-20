
// ============================================================ BiRefNet background removal
// Swin-L backbone + deformable-ASPP decoder, run with the engine's own primitives.  The matting
// weights stream in on first use and are released again before the sampler starts, so the
// background pass does not stay resident while diffusion and the decoders run.

namespace bfnet {

constexpr int kSize = 1024;

// Owns every activation allocated by a stage so no failure path can leak device memory.
struct Guard {
    IdolEngine& E;
    Tensor* list[160];
    int n = 0;
    explicit Guard(IdolEngine& e) : E(e) {}
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    Tensor& keep(Tensor& t) {
        if (n < 160) list[n++] = &t;
        return t;
    }
    ~Guard() {
        // Two slots can alias one allocation after a move-style assignment, so release each device
        // block at most once.
        CUdeviceptr done[160];
        int m = 0;
        for (int i = 0; i < n; ++i) {
            Tensor& t = *list[i];
            if (!t.p) continue;
            bool dup = false;
            for (int j = 0; j < m; ++j) {
                if (done[j] == t.p) { dup = true; break; }
            }
            if (dup) { t.p = 0; continue; }
            done[m++] = t.p;
            E.rel(t);
        }
    }
};

inline const char* dump_dir() {
    static const char* dir = std::getenv("IDOL_BFNET_DUMP");
    return dir;
}

void dump(IdolEngine& E, const char* tag, Tensor& t, int rows, int cols) {
    const char* dir = dump_dir();
    if (!dir || !t.p || rows <= 0) return;
    std::vector<float> host((size_t)rows * cols);
    if (download_f32(E, t, rows, cols, host.data())) return;
    char path[512];
    snprintf(path, sizeof path, "%s\\%s.f32", dir, tag);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(host.data(), sizeof(float), host.size(), f);
    fclose(f);
    fprintf(stderr, "[birefnet] %-8s %d x %d\n", tag, rows, cols);
}

int bn(IdolEngine& E, Tensor x, int HW, int C, const std::string& pre) {
    const std::string w = pre + ".weight", b = pre + ".bias";
    const std::string m = pre + ".running_mean", v = pre + ".running_var";
    CUdeviceptr gp = E.wget(w.c_str());
    CUdeviceptr bp = E.wget(b.c_str());
    CUdeviceptr mp = E.wget(m.c_str());
    CUdeviceptr vp = E.wget(v.c_str());
    if (!gp || !bp || !mp || !vp) return fail("birefnet: missing %s", pre.c_str());
    float eps = 1e-5f;
    void* a[] = {&x.p, &gp, &bp, &mp, &vp, &x.p, &HW, &C, &eps};
    return E.launch("k_batchnorm", dim3((HW * C + 255) / 256), dim3(256), 0, a);
}

Tensor dup(IdolEngine& E, Tensor x) {
    Tensor y = E.tmx(x.r, x.c);
    if (!y.ok()) return y;
    int n = x.r * x.c;
    void* p[] = {&y.p, &x.p, &n};
    if (E.launch("k_copy_h2h", dim3((n + 255) / 256), dim3(256), 0, p)) {
        E.rel(y);
        return Tensor{};
    }
    return y;
}

Tensor resize_nhwc(IdolEngine& E, Tensor x, int Hin, int Win, int Hout, int Wout, int C) {
    Tensor y = E.tmx(Hout * Wout, C);
    if (!y.ok()) return y;
    void* a[] = {&x.p, &y.p, &Hin, &Win, &Hout, &Wout, &C};
    int total = Hout * Wout * C;
    if (E.launch("k_resize", dim3((total + 255) / 256), dim3(256), 0, a)) {
        E.rel(y);
        return Tensor{};
    }
    return y;
}

// [a | b] along the channel axis; the padding tail is zeroed so later convolutions stay exact.
Tensor cat(IdolEngine& E, Tensor a, Tensor b, int rows) {
    int ca = a.c, cb = b.c;
    Tensor out = E.tmx(rows, (ca + cb + 63) / 64 * 64);
    if (!out.ok()) return out;
    int so = out.c;
    void* p[] = {&a.p, &ca, &b.p, &cb, &out.p, &so, &rows, &ca, &cb};
    if (E.launch("k_cat_ch", dim3((rows * so + 255) / 256), dim3(256), 0, p)) {
        E.rel(out);
        return Tensor{};
    }
    return out;
}

int swin_attention(IdolEngine& E, Tensor qkv, Tensor& out, int H, int W, int C, int heads,
                   int shift, const std::string& bias_name) {
    int Hp = (H + 11) / 12 * 12, Wp = (W + 11) / 12 * 12;
    int nwin = (Hp / 12) * (Wp / 12);
    const Weights::Entry* e = E.weights.find(bias_name.c_str());
    if (!e) return fail("birefnet: missing %s", bias_name.c_str());
    Tensor bias = E.tm(heads, 144 * 144);
    if (!bias.ok()) return fail("birefnet: oom bias");
    CUdeviceptr tp = E.wget(bias_name.c_str());
    if (!tp) {
        E.rel(bias);
        return 1;
    }
    int tstride = e->dims[1], L = 12;
    long long total = (long long)heads * 144 * 144;
    void* b[] = {&tp, &tstride, &heads, &bias.p, &L};
    if (E.launch("k_rel_bias", dim3((unsigned)((total + 255) / 256)), dim3(256), 0, b)) {
        E.rel(bias);
        return 1;
    }
    out = E.tmx(H * W, C);
    if (!out.ok()) {
        E.rel(bias);
        return fail("birefnet: oom attention");
    }
    float scale = 1.0f / std::sqrt(32.0f);
    int use_mask = shift > 0 ? 1 : 0;
    void* a[] = {&qkv.p, &out.p, &bias.p, &H, &W, &C, &heads, &Hp, &Wp, &shift, &use_mask, &scale};
    int rc = E.launch("k_swin_attn", dim3(nwin, heads), dim3(128), 0, a);
    E.rel(bias);
    if (rc) {
        E.rel(out);
        return 1;
    }
    return 0;
}

int swin_block(IdolEngine& E, Tensor& x, int H, int W, int C, int heads, int shift,
               const std::string& pre) {
    Guard g(E);
    int rows = H * W;
    Tensor hn, qkv, attn, proj, h2, f1, f2;
    g.keep(hn); g.keep(qkv); g.keep(attn); g.keep(proj); g.keep(h2); g.keep(f1); g.keep(f2);
    const char* dot = strrchr(pre.c_str(), '.');
    const bool trace = dump_dir() && H == 256 &&
                       pre.compare(0, 27, "birefnet.bb.layers.0.blocks") == 0;
    char tw[8];
    snprintf(tw, sizeof tw, "b%d", dot ? atoi(dot + 1) : 0);
    hn = dup(E, x);
    if (!hn.ok()) return 1;
    if (E.ln(hn, rows, C, (pre + ".norm1.weight").c_str(), (pre + ".norm1.bias").c_str(), 1e-5f))
        return 1;
    if (trace) dump(E, (std::string(tw) + "n1").c_str(), hn, rows, C);
    qkv = E.gemm(hn, (pre + ".attn.qkv.weight").c_str(), (pre + ".attn.qkv.bias").c_str(), 0);
    E.rel(hn);
    if (!qkv.ok()) return 1;
    if (trace) dump(E, (std::string(tw) + "qkv").c_str(), qkv, rows, qkv.c);
    if (swin_attention(E, qkv, attn, H, W, C, heads, shift,
                       pre + ".attn.relative_position_bias_table"))
        return 1;
    E.rel(qkv);
    proj = E.gemm(attn, (pre + ".attn.proj.weight").c_str(), (pre + ".attn.proj.bias").c_str(), 0);
    E.rel(attn);
    if (!proj.ok()) return 1;
    if (trace) dump(E, (std::string(tw) + "proj").c_str(), proj, rows, C);
    add_t(E, x, proj);
    E.rel(proj);
    h2 = dup(E, x);
    if (!h2.ok()) return 1;
    if (E.ln(h2, rows, C, (pre + ".norm2.weight").c_str(), (pre + ".norm2.bias").c_str(), 1e-5f))
        return 1;
    if (trace) dump(E, (std::string(tw) + "n2").c_str(), h2, rows, C);
    f1 = E.gemm(h2, (pre + ".mlp.fc1.weight").c_str(), (pre + ".mlp.fc1.bias").c_str(), 0);
    E.rel(h2);
    if (!f1.ok()) return 1;
    act_t(E, f1, 3, (int64_t)f1.r * f1.c);
    f2 = E.gemm(f1, (pre + ".mlp.fc2.weight").c_str(), (pre + ".mlp.fc2.bias").c_str(), 0);
    E.rel(f1);
    if (!f2.ok()) return 1;
    add_t(E, x, f2);
    E.rel(f2);
    if (trace) dump(E, (std::string(tw) + "out").c_str(), x, rows, C);
    return 0;
}

// Swin-L at one resolution: patch embed, four stages with patch merging, per-stage LayerNorm.
int swin(IdolEngine& E, Tensor img, int H0, int W0, Tensor out[4]) {
    Guard g(E);
    Tensor x, y, t, down;
    g.keep(x); g.keep(y); g.keep(t); g.keep(down);
    for (int i = 0; i < 4; ++i) g.keep(out[i]);
    x = conv2d(E, img, H0, W0, 3, "birefnet.bb.patch_embed.proj.weight",
               "birefnet.bb.patch_embed.proj.bias", 4, 4, 4, 0, 0, H0 / 4, W0 / 4);
    if (!x.ok()) return 1;
    int H = H0 / 4, W = W0 / 4;
    if (E.ln(x, H * W, 192, "birefnet.bb.patch_embed.norm.weight",
             "birefnet.bb.patch_embed.norm.bias", 1e-5f))
        return 1;
    if (dump_dir() && H == 256) dump(E, "emb", x, H * W, 192);
    static const int depth[4] = {2, 2, 18, 2};
    static const int heads[4] = {6, 12, 24, 48};
    for (int L = 0; L < 4; ++L) {
        int C = 192 << L;
        for (int b = 0; b < depth[L]; ++b) {
            char pre[96];
            snprintf(pre, sizeof pre, "birefnet.bb.layers.%d.blocks.%d", L, b);
            if (swin_block(E, x, H, W, C, heads[L], (b & 1) ? 6 : 0, pre)) return 1;
        }
        char nw[96], nb[96];
        snprintf(nw, sizeof nw, "birefnet.bb.norm%d.weight", L);
        snprintf(nb, sizeof nb, "birefnet.bb.norm%d.bias", L);
        y = dup(E, x);
        if (!y.ok()) return 1;
        if (E.ln(y, H * W, C, nw, nb, 1e-5f)) return 1;
        out[L] = y;
        y = Tensor{};
        if (L == 3) break;
        char dn[96], dw[128], db[128], rw[128];
        snprintf(dn, sizeof dn, "birefnet.bb.layers.%d.downsample", L);
        snprintf(dw, sizeof dw, "%s.norm.weight", dn);
        snprintf(db, sizeof db, "%s.norm.bias", dn);
        snprintf(rw, sizeof rw, "%s.reduction.weight", dn);
        int H2 = H / 2, W2 = W / 2, C4 = 4 * C;
        t = E.tmx(H2 * W2, C4);
        if (!t.ok()) return 1;
        void* a[] = {&x.p, &t.p, &H, &W, &C};
        int cnt = H2 * W2 * C4;
        if (E.launch("k_patch_merge_concat", dim3((cnt + 255) / 256), dim3(256), 0, a)) return 1;
        if (E.ln(t, H2 * W2, C4, dw, db, 1e-5f)) return 1;
        down = E.gemm(t, rw, nullptr, 0);
        E.rel(t);
        E.rel(x);
        if (!down.ok()) return 1;
        x = down;
        down = Tensor{};
        H = H2;
        W = W2;
    }
    E.rel(x);
    g.n = 0;  // out[] belongs to the caller
    return 0;
}

// ASPP deformable branch: offset/modulator heads drive a bilinear gather, then the regular kernel
// is applied as a plain GEMM over the gathered patches.
Tensor deform_branch(IdolEngine& E, Tensor x, int H, int W, int Cin, int k,
                     const std::string& pre) {
    Guard g(E);
    const std::string ob = pre + ".atrous_conv.offset_conv";
    const std::string mb = pre + ".atrous_conv.modulator_conv";
    const std::string rb = pre + ".atrous_conv.regular_conv.weight";
    int pad = k / 2;
    Tensor off, mod, out, A;
    g.keep(off); g.keep(mod); g.keep(out); g.keep(A);
    off = conv2d(E, x, H, W, Cin, (ob + ".weight").c_str(), (ob + ".bias").c_str(),
                 k, k, 1, pad, 0, H, W);
    if (!off.ok()) return Tensor{};
    mod = conv2d(E, x, H, W, Cin, (mb + ".weight").c_str(), (mb + ".bias").c_str(),
                 k, k, 1, pad, 0, H, W);
    if (!mod.ok()) return Tensor{};
    act_t(E, mod, 4, (int64_t)mod.r * mod.c);
    {
        int n = mod.r * mod.c;
        float two = 2.0f;
        void* a[] = {&mod.p, &n, &two};
        if (E.launch("k_scale", dim3((n + 255) / 256), dim3(256), 0, a)) return Tensor{};
    }
    const Weights::Entry* e = E.weights.find(rb.c_str());
    if (!e) {
        fail("birefnet: missing %s", rb.c_str());
        return Tensor{};
    }
    int Kp = e->dims[0], N = e->dims[1], T = k * k;
    CUdeviceptr wp = E.wget(rb.c_str());
    if (!wp) return Tensor{};
    out = E.tm(H * W, N);
    if (!out.ok()) return Tensor{};
    int total = H * W, CH = 1024;
    int one = 1;
    for (int p0 = 0; p0 < total; p0 += CH) {
        int n = (total - p0 < CH) ? total - p0 : CH;
        A = E.tm(n, Cin * T);
        if (!A.ok()) return Tensor{};
        if (Kp != Cin * T) E.zero(A, (int64_t)A.r * A.c);
        void* a1[] = {&x.p, &off.p, &off.c, &mod.p, &mod.c, &A.p, &A.c, &H, &W, &Cin,
                      &k, &k, &one, &pad, &H, &W, &p0, &n};
        if (E.launch("k_deform_im2col_s", dim3((n * Cin * T + 255) / 256), dim3(256), 0, a1) ||
            E.gemm_raw(A.p, A.c, n, Kp, wp, N, 0, out.p + (size_t)p0 * N * 2, N, 0))
            return Tensor{};
        E.rel(A);
    }
    if (bn(E, out, H * W, out.c, pre + ".bn")) return Tensor{};
    act_t(E, out, 2, (int64_t)out.r * out.c);
    E.rel(off);
    E.rel(mod);
    g.n = 0;  // out belongs to the caller
    return out;
}

Tensor gap_branch(IdolEngine& E, Tensor x, int H, int W, int Cin, const std::string& pre) {
    Guard g(E);
    Tensor gap, one_t, wide;
    g.keep(gap); g.keep(one_t); g.keep(wide);
    gap = E.tmx(Cin, 1);
    if (!gap.ok()) return Tensor{};
    void* a[] = {&x.p, &gap.p, &H, &W, &Cin};
    if (E.launch("k_gap", dim3((Cin + 63) / 64), dim3(64), 0, a)) return Tensor{};
    one_t = conv2d(E, gap, 1, 1, Cin, (pre + ".global_avg_pool.1.weight").c_str(), nullptr,
                   1, 1, 1, 0, 0);
    E.rel(gap);
    if (!one_t.ok()) return Tensor{};
    if (bn(E, one_t, 1, one_t.c, pre + ".global_avg_pool.2")) return Tensor{};
    act_t(E, one_t, 2, (int64_t)one_t.c);
    wide = E.tmx(H * W, one_t.c);
    if (!wide.ok()) return Tensor{};
    int rows = H * W, cols = one_t.c;
    void* b[] = {&one_t.p, &wide.p, &rows, &cols};
    if (E.launch("k_bcast_row", dim3((rows * cols + 255) / 256), dim3(256), 0, b)) return Tensor{};
    E.rel(one_t);
    g.n = 0;  // wide is handed to the caller
    return wide;
}

Tensor aspp(IdolEngine& E, Tensor x, int H, int W, int Cin, const std::string& pre) {
    static const char* kBranches[4] = {"aspp1", "aspp_deforms.0", "aspp_deforms.1", "aspp_deforms.2"};
    static const int kKernel[4] = {1, 1, 3, 7};
    Guard g(E);
    Tensor part[5], merged, out;
    for (int i = 0; i < 5; ++i) g.keep(part[i]);
    g.keep(merged); g.keep(out);
    for (int i = 0; i < 4; ++i) {
        part[i] = deform_branch(E, x, H, W, Cin, kKernel[i], pre + "." + kBranches[i]);
        if (!part[i].ok()) return Tensor{};
    }
    part[4] = gap_branch(E, x, H, W, Cin, pre);
    if (!part[4].ok()) return Tensor{};
    merged = part[0];
    part[0] = Tensor{};
    for (int i = 1; i < 5; ++i) {
        Tensor step = cat(E, merged, part[i], H * W);
        E.rel(merged);
        E.rel(part[i]);
        merged = step;
        if (!merged.ok()) return Tensor{};
    }
    out = conv2d(E, merged, H, W, merged.c, (pre + ".conv1.weight").c_str(), nullptr,
                 1, 1, 1, 0, 0);
    E.rel(merged);
    if (!out.ok()) return Tensor{};
    if (bn(E, out, H * W, out.c, pre + ".bn1")) return Tensor{};
    act_t(E, out, 2, (int64_t)out.r * out.c);
    g.n = 0;
    return out;
}

// conv/BN/ReLU -> deformable ASPP -> conv/BN (no trailing ReLU, matching _BasicDecBlk).
Tensor dec_block(IdolEngine& E, Tensor x, int H, int W, int Cin, const std::string& pre) {
    Guard g(E);
    Tensor c, a, o;
    g.keep(c); g.keep(a); g.keep(o);
    c = conv2d(E, x, H, W, Cin, (pre + ".conv_in.weight").c_str(), (pre + ".conv_in.bias").c_str(),
               3, 3, 1, 1, 0);
    if (!c.ok()) return Tensor{};
    if (bn(E, c, H * W, c.c, pre + ".bn_in")) return Tensor{};
    act_t(E, c, 2, (int64_t)c.r * c.c);
    a = aspp(E, c, H, W, c.c, pre + ".dec_att");
    E.rel(c);
    if (!a.ok()) return Tensor{};
    o = conv2d(E, a, H, W, a.c, (pre + ".conv_out.weight").c_str(), (pre + ".conv_out.bias").c_str(),
               3, 3, 1, 1, 0);
    E.rel(a);
    if (!o.ok()) return Tensor{};
    if (bn(E, o, H * W, o.c, pre + ".bn_out")) return Tensor{};
    g.n = 0;
    return o;
}

Tensor simple_convs(IdolEngine& E, Tensor x, int H, int W, int Cin, const std::string& pre) {
    Guard g(E);
    Tensor c, o;
    g.keep(c); g.keep(o);
    c = conv2d(E, x, H, W, Cin, (pre + ".conv1.weight").c_str(), (pre + ".conv1.bias").c_str(),
               3, 3, 1, 1, 0);
    if (!c.ok()) return Tensor{};
    o = conv2d(E, c, H, W, c.c, (pre + ".conv_out.weight").c_str(), (pre + ".conv_out.bias").c_str(),
               3, 3, 1, 1, 0);
    E.rel(c);
    if (!o.ok()) return Tensor{};
    g.n = 0;
    return o;
}

int gdt_gate(IdolEngine& E, Tensor& p, int H, int W, const std::string& conv_pre,
             const std::string& attn_pre) {
    Guard g(E);
    Tensor g1, gate;
    g.keep(g1); g.keep(gate);
    g1 = conv2d(E, p, H, W, p.c, (conv_pre + ".0.weight").c_str(), (conv_pre + ".0.bias").c_str(),
                3, 3, 1, 1, 0);
    if (!g1.ok()) return 1;
    if (bn(E, g1, H * W, g1.c, conv_pre + ".1")) return 1;
    act_t(E, g1, 2, (int64_t)g1.r * g1.c);
    gate = conv2d(E, g1, H, W, g1.c, (attn_pre + ".weight").c_str(), (attn_pre + ".bias").c_str(),
                  1, 1, 1, 0, 0);
    E.rel(g1);
    if (!gate.ok()) return 1;
    int rows = H * W, cols = p.c, gs = gate.c;
    void* args[] = {&p.p, &gate.p, &gs, &rows, &cols};
    return E.launch("k_gate_mul", dim3((rows * cols + 255) / 256), dim3(256), 0, args);
}

Tensor image2patches(IdolEngine& E, Tensor img, int Hf, int Wf, int Hr, int Wr, int C) {
    int hg = Hf / Hr, wg = Wf / Wr;
    Tensor t = E.tmx(Hr * Wr, C * hg * wg);
    if (!t.ok()) return t;
    void* a[] = {&img.p, &t.p, &Hf, &Wf, &Hr, &Wr, &C};
    int total = Hr * Wr * C * hg * wg;
    if (E.launch("k_image2patches", dim3((total + 255) / 256), dim3(256), 0, a)) {
        E.rel(t);
        return Tensor{};
    }
    return t;
}

// image -> patch stack -> the decoder's two-conv input adapter (ipt_blk*).
Tensor ipt_block(IdolEngine& E, Tensor img, int Hr, int Wr, const std::string& pre) {
    Guard g(E);
    Tensor raw, out;
    g.keep(raw);
    g.keep(out);
    raw = image2patches(E, img, kSize, kSize, Hr, Wr, 3);
    if (!raw.ok()) return Tensor{};
    out = simple_convs(E, raw, Hr, Wr, raw.c, pre);
    if (!out.ok()) return Tensor{};
    g.n = 0;
    return out;
}

// Decoder half: x[0..3] are the mul-scale-cat features at 256/128/64/32.
int decode(IdolEngine& E, Tensor img, Tensor x[4], Tensor& alpha_out) {
    Guard g(E);
    Tensor x4c, sq, ipt, x4d, p, cur, up, lat, merged, full, tail, logits;
    g.keep(x4c); g.keep(sq); g.keep(ipt); g.keep(x4d); g.keep(p); g.keep(cur); g.keep(up);
    g.keep(lat); g.keep(merged); g.keep(full); g.keep(tail); g.keep(logits);
    int S = kSize;
    x4c = resize_nhwc(E, x[0], 256, 256, 32, 32, x[0].c);
    if (!x4c.ok()) return 1;
    for (int i = 1; i < 4; ++i) {
        int sz = 256 >> i;
        up = resize_nhwc(E, x[i], sz, sz, 32, 32, x[i].c);
        if (!up.ok()) return 1;
        merged = cat(E, x4c, up, 32 * 32);
        E.rel(x4c);
        E.rel(up);
        x4c = merged;
        merged = Tensor{};
        if (!x4c.ok()) return 1;
    }
    dump(E, "x4c", x4c, 32 * 32, x4c.c);
    sq = dec_block(E, x4c, 32, 32, x4c.c, "birefnet.squeeze_module.0");
    E.rel(x4c);
    if (!sq.ok()) return 1;
    dump(E, "squeeze", sq, 32 * 32, sq.c);
    ipt = ipt_block(E, img, 32, 32, "birefnet.decoder.ipt_blk5");
    if (!ipt.ok()) return 1;
    dump(E, "ipt5", ipt, 32 * 32, ipt.c);
    x4d = cat(E, sq, ipt, 32 * 32);
    E.rel(sq);
    E.rel(ipt);
    if (!x4d.ok()) return 1;
    p = dec_block(E, x4d, 32, 32, x4d.c, "birefnet.decoder.decoder_block4");
    E.rel(x4d);
    if (!p.ok()) return 1;
    dump(E, "p4_raw", p, 32 * 32, p.c);
    if (gdt_gate(E, p, 32, 32, "birefnet.decoder.gdt_convs_4",
                 "birefnet.decoder.gdt_convs_attn_4.0"))
        return 1;
    dump(E, "p4", p, 32 * 32, p.c);
    cur = resize_nhwc(E, p, 32, 32, 64, 64, p.c);
    E.rel(p);
    if (!cur.ok()) return 1;
    static const int kSize2[3] = {64, 128, 256};
    for (int s = 0; s < 3; ++s) {
        int Hs = kSize2[s];
        char latn[96], decn[96], iptn[96], gdn[96], gan[96];
        snprintf(latn, sizeof latn, "birefnet.decoder.lateral_block%d.conv", 4 - s);
        snprintf(decn, sizeof decn, "birefnet.decoder.decoder_block%d", 3 - s);
        snprintf(iptn, sizeof iptn, "birefnet.decoder.ipt_blk%d", 4 - s);
        snprintf(gdn, sizeof gdn, "birefnet.decoder.gdt_convs_%d", 3 - s);
        snprintf(gan, sizeof gan, "birefnet.decoder.gdt_convs_attn_%d.0", 3 - s);
        // lateral_block4 pairs with x3 (64), lateral_block3 with x2 (128), lateral_block2 with x1 (256).
        Tensor& xf = x[2 - s];
        lat = conv2d(E, xf, Hs, Hs, xf.c, (std::string(latn) + ".weight").c_str(),
                     (std::string(latn) + ".bias").c_str(), 1, 1, 1, 0, 0);
        if (!lat.ok()) return 1;
        add_t(E, cur, lat);
        E.rel(lat);
        {
            char tag[16];
            snprintf(tag, sizeof tag, "_p%d", 3 - s);
            dump(E, tag, cur, Hs * Hs, cur.c);
        }
        ipt = ipt_block(E, img, Hs, Hs, iptn);
        if (!ipt.ok()) return 1;
        {
            char tag[16];
            snprintf(tag, sizeof tag, "ipt%d", 4 - s);
            dump(E, tag, ipt, Hs * Hs, ipt.c);
        }
        merged = cat(E, cur, ipt, Hs * Hs);
        E.rel(cur);
        E.rel(ipt);
        if (!merged.ok()) return 1;
        p = dec_block(E, merged, Hs, Hs, merged.c, decn);
        E.rel(merged);
        if (!p.ok()) return 1;
        if (s == 2) {
            dump(E, "dec1", p, 256 * 256, p.c);
            cur = p;
            p = Tensor{};
            break;
        }
        {
            char tag[16];
            snprintf(tag, sizeof tag, "p%draw", 3 - s);
            dump(E, tag, p, Hs * Hs, p.c);
        }
        if (gdt_gate(E, p, Hs, Hs, gdn, gan)) return 1;
        char tag[16];
        snprintf(tag, sizeof tag, "p%d", 3 - s);
        dump(E, tag, p, Hs * Hs, p.c);
        up = resize_nhwc(E, p, Hs, Hs, Hs * 2, Hs * 2, p.c);
        E.rel(p);
        if (!up.ok()) return 1;
        cur = up;
        up = Tensor{};
    }
    E.pool.trim();  // the 1024x1024 maps dominate the footprint; start them from a clean pool
    full = resize_nhwc(E, cur, 256, 256, S, S, cur.c);
    E.rel(cur);
    if (!full.ok()) return 1;
    ipt = ipt_block(E, img, S, S, "birefnet.decoder.ipt_blk1");
    if (!ipt.ok()) return 1;
    dump(E, "ipt1", ipt, S * S, ipt.c);
    tail = cat(E, full, ipt, S * S);
    E.rel(full);
    E.rel(ipt);
    if (!tail.ok()) return 1;
    logits = conv2d(E, tail, S, S, tail.c, "birefnet.decoder.conv_out1.0.weight",
                    "birefnet.decoder.conv_out1.0.bias", 1, 1, 1, 0, 0);
    E.rel(tail);
    if (!logits.ok()) return 1;
    dump(E, "logits", logits, S * S, 1);
    alpha_out = E.tmx(S * S, 1);
    if (!alpha_out.ok()) return 1;
    g.keep(alpha_out);
    {
        int rows = S * S, cols = 1;
        void* ap[] = {&logits.p, &logits.c, &alpha_out.p, &rows, &cols};
        if (E.launch("k_sigmoid_strided", dim3((rows + 255) / 256), dim3(256), 0, ap)) return 1;
    }
    E.rel(logits);
    dump(E, "alpha", alpha_out, S * S, 1);
    g.n = 0;  // alpha_out belongs to the caller
    return 0;
}

int encode(IdolEngine& E, Tensor img, Tensor x[4]) {
    Guard g(E);
    Tensor a[4], b[4], half, up, merged;
    for (int i = 0; i < 4; ++i) { g.keep(a[i]); g.keep(b[i]); }
    g.keep(half); g.keep(up); g.keep(merged);
    for (int i = 0; i < 4; ++i) g.keep(x[i]);
    if (swin(E, img, kSize, kSize, a)) return 1;
    static const char* kTags[4] = {"x1a", "x2a", "x3a", "x4a"};
    static const int kSpat[4] = {256, 128, 64, 32};
    for (int i = 0; i < 4; ++i) dump(E, kTags[i], a[i], kSpat[i] * kSpat[i], a[i].c);
    half = resize_nhwc(E, img, kSize, kSize, kSize / 2, kSize / 2, 3);
    if (!half.ok()) return 1;
    int rc = swin(E, half, kSize / 2, kSize / 2, b);
    E.rel(half);
    if (rc) return 1;
    static const char* kTagsB[4] = {"x1b", "x2b", "x3b", "x4b"};
    static const int kSpatB[4] = {128, 64, 32, 16};
    for (int i = 0; i < 4; ++i) dump(E, kTagsB[i], b[i], kSpatB[i] * kSpatB[i], b[i].c);
    for (int i = 0; i < 4; ++i) {
        up = resize_nhwc(E, b[i], kSpatB[i], kSpatB[i], kSpat[i], kSpat[i], b[i].c);
        if (!up.ok()) return 1;
        merged = cat(E, a[i], up, kSpat[i] * kSpat[i]);
        E.rel(up);
        E.rel(a[i]);
        E.rel(b[i]);
        if (!merged.ok()) return 1;
        x[i] = merged;
        merged = Tensor{};
    }
    static const char* kTagsX[4] = {"x1", "x2", "x3", "x4"};
    for (int i = 0; i < 4; ++i) dump(E, kTagsX[i], x[i], kSpat[i] * kSpat[i], x[i].c);
    g.n = 0;  // x[] belongs to the caller
    return 0;
}

}  // namespace bfnet

int birefnet_mask(IdolEngine& E, const float* rgba, int w, int h, float* alpha) {
    using namespace bfnet;
    int S = kSize;
    Tensor img = E.tmx(S * S, 3);
    if (!img.ok()) return fail("birefnet: oom image");
    size_t bytes = (size_t)w * h * 4 * sizeof(float);
    CUdeviceptr host = E.pool.alloc(bytes);
    if (!host) {
        E.rel(img);
        return fail("birefnet: oom image upload");
    }
    if (E.drv.htod(host, rgba, bytes) != CUDA_SUCCESS) {
        E.pool.release(host, bytes);
        E.rel(img);
        return fail("birefnet: image upload failed");
    }
    float s0 = 1.0f / 0.229f, s1 = 1.0f / 0.224f, s2 = 1.0f / 0.225f;
    float b0 = -0.485f / 0.229f, b1 = -0.456f / 0.224f, b2 = -0.406f / 0.225f;
    {
        int cnt = S * S * 3;
        void* a[] = {&host, &img.p, &h, &w, &S, &S, &s0, &s1, &s2, &b0, &b1, &b2};
        int rc = E.launch("k_img_prep", dim3((cnt + 255) / 256), dim3(256), 0, a);
        E.pool.release(host, bytes);
        if (rc) {
            E.rel(img);
            return 1;
        }
    }
    Tensor x[4];
    if (encode(E, img, x)) {
        E.rel(img);
        return 1;
    }
    E.pool.trim();  // hand the encoder's scratch back before the decoder allocates
    Tensor packed;
    int rc = decode(E, img, x, packed);
    E.rel(img);
    for (int i = 0; i < 4; ++i) E.rel(x[i]);
    if (rc) return 1;
    Tensor out = E.tmx(w * h, 1);
    if (!out.ok()) {
        E.rel(packed);
        return fail("birefnet: oom alpha");
    }
    {
        int one = 1;
        void* a[] = {&packed.p, &out.p, &S, &S, &h, &w, &one};
        int r2 = E.launch("k_resize", dim3((w * h + 255) / 256), dim3(256), 0, a);
        E.rel(packed);
        if (r2) {
            E.rel(out);
            return 1;
        }
    }
    int rc2 = download_f32(E, out, h * w, 1, alpha);
    E.rel(out);
    E.pool.trim();
    E.free_weights_prefix("birefnet.");
    return rc2;
}

