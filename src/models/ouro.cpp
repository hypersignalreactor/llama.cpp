#include "models.h"
#include "ouro-adaptive.h"

#include "llama-kv-cache.h"

#include <cstdio>
#include <cstdlib>

ouro_stage_state & ouro_stage() {
    static ouro_stage_state st;
    return st;
}

static void ouro_print_stats() {
    const auto & st = ouro_stage();
    int64_t n = 0, sum = 0;
    for (int i = 0; i < st.n_loops && i < 16; ++i) { n += st.n_exit[i]; sum += st.n_exit[i] * (i + 1); }
    if (n == 0) { return; }
    fprintf(stderr, "ouro adaptive depth (p >= %.2f): %lld tokens, %.2f loops on average of %d; stopped after loop", st.exit_p, (long long) n, (double) sum / n, st.n_loops);
    for (int i = 0; i < st.n_loops && i < 16; ++i) { fprintf(stderr, " %d: %lld", i + 1, (long long) st.n_exit[i]); }
    fprintf(stderr, "\n");
}

// the hidden state carried from the previous loop (staged decoding)
class llm_graph_input_ouro_carry : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch *) override {
        const auto & st = ouro_stage();
        GGML_ASSERT(st.carry.size() * sizeof(float) == ggml_nbytes(t));
        ggml_backend_tensor_set(t, st.carry.data(), 0, ggml_nbytes(t));
    }
    bool can_reuse(const llm_graph_params &) override { return false; }
    ggml_tensor * t = nullptr;
};

// Ouro (ByteDance Seed) — a looped / universal-transformer LM: the whole decoder stack is applied
// `num_loops` times per token with shared weights. Reference: modeling_ouro.py, OuroModel.forward,
//
//     for current_ut in range(total_ut_steps):
//         for layer in layers: hidden = layer(hidden, current_ut=current_ut)
//         hidden = self.norm(hidden)          # every loop, including the last
//
// and each layer caches K/V at index `current_ut * num_hidden_layers + layer_idx`, so every
// (loop, layer) pair owns an independent cache slot. That is exactly the unroll already used for
// nanbeige: expand the logical layer count to n_layer_phys * n_loops, share the physical weights
// across loops, and let each logical layer keep its own KV index.
//
// Ouro's block differs from a Llama block by using sandwich norms — a norm on the attention output
// and on the FFN output, before each residual add (modeling_ouro.py, OuroDecoderLayer.forward):
//
//     h = x + input_layernorm_2(attn(input_layernorm(x)))
//     h = h + post_attention_layernorm_2(mlp(post_attention_layernorm(h)))
//
// which maps onto the existing ATTN_POST_NORM / FFN_POST_NORM tensors.

void llama_model_ouro::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    uint32_t n_loops_u = 1;
    ml.get_key(LLM_KV_NUM_LOOPS, n_loops_u, false);
    GGML_ASSERT(n_loops_u >= 1);

    n_layer_phys = (int) hparams.n_layer();

    // Bound-check before casting: signed int mul can overflow and bypass the guard.
    GGML_ASSERT((size_t) n_layer_phys * (size_t) n_loops_u <= (size_t) LLAMA_MAX_LAYERS);
    n_loops = (int) n_loops_u;

    {
        auto & st = ouro_stage();
        st.n_loops = n_loops;
        const char * ep = getenv("OURO_EXIT_P");
        const char * ml = getenv("OURO_MIN_LOOPS");
        st.exit_p    = ep ? (float) atof(ep) : 0.0f;
        st.min_loops = ml ? atoi(ml) : 1;
        const char * sp = getenv("OURO_STEP");
        const char * fl = getenv("OURO_FILL");
        st.step = sp ? atoi(sp) : 1;
        st.fill = fl && fl[0] == 's' ? 1 : 0;
        if (st.min_loops < 1) { st.min_loops = 1; }
        if (st.step < 1) { st.step = 1; }
        if (st.exit_p > 0.0f && n_loops > 1) {
            static bool once = false;
            if (!once) { once = true; atexit(ouro_print_stats); }
        }
    }

    // Expand logical layer count before load_tensors() allocates layers / KV.
    if (n_loops > 1) {
        for (int j = 1; j < n_loops; ++j) {
            for (int i = 0; i < n_layer_phys; ++i) {
                const int dst = i + j * n_layer_phys;
                hparams.n_head_arr[dst]    = hparams.n_head_arr[i];
                hparams.n_head_kv_arr[dst] = hparams.n_head_kv_arr[i];
                hparams.n_ff_arr[dst]      = hparams.n_ff_arr[i];
                hparams.is_swa_impl[dst]   = hparams.is_swa_impl[i];
                hparams.is_recr_impl[dst]  = hparams.is_recr_impl[i];
            }
        }
        hparams.n_layer_all = (uint32_t) ((size_t) n_layer_phys * (size_t) n_loops);
        // -ngl counts the physical layers, the loops share their weights
        hparams.n_layer_shared = (uint32_t) n_layer_phys;
    }

    switch (n_layer_phys) {
        case 24: type = LLM_TYPE_1_4B; break;
        case 48: type = LLM_TYPE_2_6B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_ouro::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    const int n_phys = n_layer_phys > 0 ? n_layer_phys : n_layer;
    for (int i = 0; i < n_phys; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // sandwich norms: applied to the sublayer output, before the residual add
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }

    // Apollos fork: the early-exit gate (model.early_exit_gate, Linear(n_embd, 1)), stored as cls.output when present.
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {1},         TENSOR_NOT_REQUIRED);

    // Share physical weights across loops; each slot still has its own KV index.
    if (n_loops > 1) {
        for (int j = 1; j < n_loops; ++j) {
            for (int i = 0; i < n_phys; ++i) {
                layers[i + j * n_phys] = layers[i];
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_ouro::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_ouro::graph::graph(const llm_graph_params & params) :
        llm_graph_context(params) {
    const auto & ou = static_cast<const llama_model_ouro &>(model);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int n_phys  = ou.n_layer_phys > 0 ? ou.n_layer_phys : (int) n_layer;
    const int n_loops = ou.n_loops > 0 ? ou.n_loops : 1;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // staged decoding (see ouro-adaptive.h): only loop `stage` is built, fed by the previous loop's normed hidden state
    const int  s_lo   = ouro_stage().lo;
    const int  s_hi   = ouro_stage().hi;
    const bool staged = s_lo >= 0;
    const int  il_beg = staged ? s_lo * n_phys : 0;
    const int  il_end = staged ? s_hi * n_phys : (int) n_layer;
    const bool fill_copy  = staged && s_hi < n_loops && ouro_stage().fill == 0;
    const bool fill_state = staged && s_hi < n_loops && ouro_stage().fill == 1;

    if (s_lo > 0) {
        auto inp = std::make_unique<llm_graph_input_ouro_carry>();
        inp->t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->t);
        ggml_set_name(inp->t, "ouro_carry");
        inpL = inp->t;
        res->add_input(std::move(inp));
    } else {
        inpL = build_inp_embd(model.tok_embd);
    }

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = staged ? nullptr : build_inp_out_ids();

    // Apollos fork, probe mode (OURO_PROBE=1): expose every loop's exit-gate logit and LM-head logits as named nodes
    // ("ouro_gate-<loop>", "ouro_logits-<loop>") so a tool can read them through cb_eval. All positions must be outputs.
    const bool probe = getenv("OURO_PROBE") != nullptr;
    auto tap = [&](ggml_tensor * h, int loop) {
        if (!probe) { return; }
        if (model.cls_out) {
            ggml_tensor * g = ggml_mul_mat(ctx0, model.cls_out, h);
            if (model.cls_out_b) { g = ggml_add(ctx0, g, model.cls_out_b); }
            ggml_format_name(g, "ouro_gate-%d", loop);
            ggml_build_forward_expand(gf, g);
        }
        ggml_tensor * lg = build_lora_mm(model.output, h, model.output_s);
        ggml_format_name(lg, "ouro_logits-%d", loop);
        ggml_build_forward_expand(gf, lg);
    };

    for (int il = il_beg; il < il_end; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // copy the stage's last loop's K/V forward into every later loop's slot for these cells
            if (fill_copy && il >= (s_hi - 1) * n_phys) {
                for (int j = s_hi; j < n_loops; ++j) {
                    const int il2 = il + (j - (s_hi - 1)) * n_phys;
                    ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il2));
                    ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il2));
                }
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // norm on the attention output, before the residual add
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // norm on the FFN output, before the residual add
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;

        // The reference applies model.norm at the end of every loop, and the normed state is what
        // feeds the next loop. The final loop's norm is the shared one applied after this loop.
        if (n_loops > 1 && ((il + 1) % n_phys) == 0 && (il + 1) < il_end) {
            cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "loop_norm", il);
            inpL = cur;
            tap(cur, (il + 1) / n_phys);
        }
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    tap(cur, n_loops);

    // state fill: every later loop's K/V for these cells, projected from this stage's final hidden state
    if (fill_state) {
        for (int i = 0; i < n_phys; ++i) {
            ggml_tensor * x = build_norm(cur, model.layers[i].attn_norm, NULL, LLM_NORM_RMS, i);
            auto [Qf, Kf, Vf] = build_qkv(model.layers[i], x, n_embd_head, n_head, n_head_kv, i);
            (void) Qf;
            Kf = build_rope(Kf, inp_pos, i);
            for (int j = s_hi; j < n_loops; ++j) {
                const int il2 = i + j * n_phys;
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kf, inp_attn->get_k_idxs(), il2));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vf, inp_attn->get_v_idxs(), il2));
            }
        }
    }

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
