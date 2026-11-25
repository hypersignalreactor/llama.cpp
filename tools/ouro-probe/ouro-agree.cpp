// ouro-agree: how faithful is a decoding mode to full depth, token by token?
//   gen    greedy-generate n tokens after each prompt, one token per decode; write the token ids to <ref>
//   force  read <ref>; feed its tokens one per decode and count how often this mode's top-1 equals the reference token
// Prompts are separated by a line holding only "---". Run gen without OURO_EXIT_P for the reference.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: llama-ouro-agree model.gguf prompts.txt ref.txt gen|force n_predict [n_gpu_layers]\n"); return 1; }
    const bool force = strcmp(argv[4], "force") == 0; const int n_pred = atoi(argv[5]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = argc > 6 ? atoi(argv[6]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model); const int nv = llama_vocab_n_tokens(vocab);
    auto cp = llama_context_default_params(); cp.n_ctx = 1024; cp.n_batch = 1024;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::ifstream f(argv[2]); std::vector<std::string> prompts; std::string line, cur;
    while (std::getline(f, line)) { if (line == "---") { prompts.push_back(cur); cur.clear(); } else { cur += (cur.empty() ? "" : "\n") + line; } }
    if (!cur.empty()) { prompts.push_back(cur); }
    std::vector<std::vector<llama_token>> ref;
    if (force) { std::ifstream r(argv[3]); while (std::getline(r, line)) { std::istringstream is(line); std::vector<llama_token> v; int t; while (is >> t) { v.push_back(t); } ref.push_back(v); } }
    FILE * out = force ? nullptr : fopen(argv[3], "w");
    long n_tok = 0, n_same = 0; double ms = 0;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const std::string & p = prompts[pi];
        std::vector<llama_token> toks(p.size() + 8); int n = llama_tokenize(vocab, p.c_str(), p.size(), toks.data(), toks.size(), true, false); toks.resize(n);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n))) { fprintf(stderr, "decode failed\n"); return 1; }
        std::string text;
        for (int i = 0; i < n_pred; i++) {
            const float * l = llama_get_logits_ith(ctx, -1); int am = 0; for (int v = 1; v < nv; v++) { if (l[v] > l[am]) { am = v; } }
            llama_token next = am;
            if (force) { if (pi >= ref.size() || i >= (int) ref[pi].size()) { break; } n_same += am == ref[pi][i]; next = ref[pi][i]; }
            else { fprintf(out, "%d ", am); }
            n_tok++;
            char buf[64]; int k = llama_token_to_piece(vocab, force ? am : next, buf, sizeof(buf), 0, false); if (k > 0) { text.append(buf, k); }
            const auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (out) { fprintf(out, "\n"); }
        std::replace(text.begin(), text.end(), '\n', ' ');
        printf("[%zu] %s\n", pi, text.c_str());
    }
    if (out) { fclose(out); }
    printf("RESULT mode=%s tokens=%ld ms_per_token=%.1f", argv[4], n_tok, ms / std::max(1L, n_tok));
    if (force) { printf(" agree=%.2f%% (%ld of %ld)", 100.0 * n_same / std::max(1L, n_tok), n_same, n_tok); }
    printf("\n"); fflush(stdout);
    return 0;
}
