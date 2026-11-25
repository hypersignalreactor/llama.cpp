// ouro-probe: teacher-forced pass over a text; for every position and every loop, what the model would predict if it
// stopped there (top-1, its probability, entropy, probability of the true next token) and the exit gate's logit.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
struct Cap { std::map<std::string, std::vector<float>> t; };
static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    bool want = strncmp(t->name, "ouro_gate-", 10) == 0 || strncmp(t->name, "ouro_logits-", 12) == 0;
    if (ask) return want;
    if (want) { auto & v = ((Cap *) ud)->t[t->name]; v.resize(ggml_nelements(t)); ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t)); }
    return true;
}
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: probe model.gguf text.txt out.csv [chunk=256]\n"); return 1; }
    const int chunk = argc > 4 ? atoi(argv[4]) : 256;
    llama_backend_init();
    auto mp = llama_model_default_params(); llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model); const int nv = llama_vocab_n_tokens(vocab);
    Cap cap; auto cp = llama_context_default_params(); cp.n_ctx = chunk; cp.n_batch = chunk; cp.n_ubatch = chunk; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.cb_eval = cb; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 8); int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false); toks.resize(n);
    FILE * out = fopen(argv[3], "w"); fprintf(out, "chunk,pos,tok,next");
    for (int k = 1; k <= 4; k++) fprintf(out, ",top%d,ptop%d,ent%d,ptrue%d,gate%d", k, k, k, k, k); fprintf(out, "\n");
    for (int c0 = 0, ci = 0; c0 + 8 < n; c0 += chunk, ci++) {
        int m = std::min(chunk, n - c0); llama_memory_clear(llama_get_memory(ctx), true); cap.t.clear();
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; i++) { b.token[i] = toks[c0 + i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; } b.n_tokens = m;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 0; i + 1 < m; i++) {
            fprintf(out, "%d,%d,%d,%d", ci, i, toks[c0 + i], toks[c0 + i + 1]);
            for (int k = 1; k <= 4; k++) {
                auto it = cap.t.find("ouro_logits-" + std::to_string(k)); if (it == cap.t.end()) { fprintf(out, ",,,,,"); continue; }
                const float * l = it->second.data() + (size_t) i * nv; float mx = l[0]; int am = 0; for (int v = 1; v < nv; v++) if (l[v] > mx) { mx = l[v]; am = v; }
                double z = 0; for (int v = 0; v < nv; v++) z += exp((double) l[v] - mx); double lz = log(z), ent = 0;
                for (int v = 0; v < nv; v++) { double lp = l[v] - mx - lz; double p = exp(lp); if (p > 1e-12) ent -= p * lp; }
                auto g = cap.t.find("ouro_gate-" + std::to_string(k));
                fprintf(out, ",%d,%.5f,%.4f,%.6g,%.4f", am, exp(-lz), ent, exp((double) l[toks[c0 + i + 1]] - mx - lz), g == cap.t.end() ? NAN : g->second[i]);
            }
            fprintf(out, "\n");
        }
        llama_batch_free(b); fprintf(stderr, "chunk %d (%d tokens)\n", ci, m);
    }
    fclose(out); return 0;
}
