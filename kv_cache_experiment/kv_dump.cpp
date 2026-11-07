// kv_dump.cpp
// Load a Gemma-3 GGUF, apply the chat template, generate a fixed number of
// tokens with a greedy sampler on CPU, then save the KV/session state with
// llama_state_save_file so we can inspect the raw attention K/V tensors.
//
// Build:
//   g++ -O2 -std=c++17 kv_dump.cpp -I$LLAMA/include \
//     -L$LLAMA/build/bin -Wl,-rpath,$LLAMA/build/bin \
//     -llama -lggml -lggml-cpu -lggml-base -o kv_dump
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * kPrompt =
    "a pony's first day at its office job as a political consultant. "
    "something odd and complicated";

int main(int argc, char ** argv) {
    const char * model_path   = (argc > 1) ? argv[1] :
        "/home/bigattichouse/workspace/gguf/gemma-3-270m-it-Q8_0.gguf";
    const char * session_path = (argc > 2) ? argv[2] : "kv_session.ggsn";
    const int n_gen = (argc > 3) ? atoi(argv[3]) : 512;

    printf("== kv_dump ==\n");
    printf("model   : %s\n", model_path);
    printf("session : %s\n", session_path);
    printf("n_gen   : %d\n", n_gen);
    fflush(stdout);

    // ---- load model (CPU only) ----
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;             // keep everything on CPU

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { printf("FATAL: failed to load model\n"); return 1; }
    printf("model loaded. n_layer = %d\n", llama_model_n_layer(model));
    fflush(stdout);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = 2048;
    cparams.n_batch       = 512;
    cparams.n_threads     = 4;
    cparams.n_threads_batch = 4;
    cparams.no_perf       = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { printf("FATAL: failed to init context\n"); return 1; }
    printf("context ready. n_ctx = %u\n", cparams.n_ctx);
    fflush(stdout);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- build chat prompt (Gemma template) ----
    // A tmpl string containing "<start_of_turn>" routes to the Gemma path.
    std::vector<struct llama_chat_message> chat;
    chat.push_back({ "user", kPrompt });

    const char * tmpl = "<start_of_turn>";
    std::string prompt_text;
    int need = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                         /*add_ass=*/true, nullptr, 0);
    if (need < 0) { printf("FATAL: template not supported\n"); return 1; }
    prompt_text.resize(need + 1);
    int got = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                        /*add_ass=*/true,
                                        &prompt_text[0], (int32_t) prompt_text.size());
    prompt_text.resize(got);
    printf("prompt text (%d bytes):\n%.*s\n", got, got, prompt_text.data());
    fflush(stdout);

    // ---- tokenize ----
    const int32_t n_tok_max = 1024;
    std::vector<llama_token> tokens(n_tok_max);
    int n_tok = llama_tokenize(vocab, prompt_text.c_str(), (int32_t) prompt_text.size(),
                               tokens.data(), n_tok_max,
                               /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) { printf("FATAL: tokenize failed\n"); return 1; }
    tokens.resize(n_tok);
    printf("tokenized prompt: %d tokens\n", n_tok);
    for (int i = 0; i < n_tok; i++) printf("  %d ", tokens[i]);
    printf("\n");
    fflush(stdout);

    // ---- decode prompt (compute logits only at the last position) ----
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
        // get_one leaves logits=nullptr; allocate and request logits at the
        // last token so the sampler has something to read.
        batch.logits = (int8_t *) malloc(sizeof(int8_t) * n_tok);
        for (int i = 0; i < n_tok; i++) batch.logits[i] = (i == n_tok - 1) ? 1 : 0;
        int rc = llama_decode(ctx, batch);
        batch.token = nullptr;           // get_one wraps our external buffer; don't free it
        llama_batch_free(batch);          // frees our malloc'd logits
        if (rc != 0) { printf("FATAL: prompt decode rc=%d\n", rc); return 1; }
    }

    // ---- greedy generation ----
    llama_sampler * smpl = llama_sampler_init_greedy();
    std::vector<llama_token> generated;
    // idx here is the *batch position* whose logits were requested. We set
    // logits only at the last position of the prompt batch, so sample at n_tok-1.
    llama_token id = llama_sampler_sample(smpl, ctx, /*idx=*/n_tok - 1);
    generated.push_back(id);
    while ((int) generated.size() < n_gen && !llama_vocab_is_eog(vocab, id)) {
        llama_batch batch = llama_batch_get_one(&id, 1);
        batch.logits = (int8_t *) malloc(sizeof(int8_t));
        batch.logits[0] = 1;  // need logits here to sample the next token
        int rc = llama_decode(ctx, batch);
        batch.token = nullptr;           // get_one wraps our local; don't free it
        llama_batch_free(batch);          // frees our malloc'd logits
        if (rc != 0) { printf("FATAL: gen decode rc=%d at tok %zu\n", rc, generated.size()); return 1; }
        id = llama_sampler_sample(smpl, ctx, 0);
        generated.push_back(id);
    }
    printf("generated %zu tokens\n", generated.size());
    fflush(stdout);

    // ---- print generated text (first ~600 chars) ----
    {
        std::string text;
        for (size_t i = 0; i < generated.size(); i++) {
            const char * t = llama_vocab_get_text(vocab, generated[i]);
            if (t) text += t;
        }
        printf("generated text (%zu chars):\n%s\n", text.size(), text.c_str());
    }
    fflush(stdout);

    // ---- save full session state (prompt + generated) ----
    std::vector<llama_token> all = tokens;
    for (auto t : generated) all.push_back(t);
    bool ok = llama_state_save_file(ctx, session_path, all.data(), all.size());
    printf("session saved: %s (ok=%d, total %zu tokens)\n",
           session_path, (int) ok, all.size());
    fflush(stdout);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    printf("DONE\n");
    return 0;
}
