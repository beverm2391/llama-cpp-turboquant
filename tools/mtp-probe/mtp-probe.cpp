#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-ext.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static llama_token argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float bestv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bestv) {
            bestv = logits[v];
            best  = v;
        }
    }
    return best;
}

static bool context_trim(llama_context * ctx, llama_pos p0) {
    llama_memory_t mem = llama_get_memory(ctx);
    return mem != nullptr && llama_memory_seq_rm(mem, 0, p0, -1);
}

int main(int argc, char ** argv) {
    common_params params;
    params.prompt = "The history of the Roman Empire is a long and complex story that begins with";
    params.n_predict = 128;
    params.warmup = false;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 1;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    params.warmup = false;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.n_parallel = 1;
    if (params.n_predict <= 0) {
        LOG_ERR("%s: --predict must be positive\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    int result = 0;
    {
        common_init_result_ptr llama_init = common_init_from_params(params);
        llama_model * model = llama_init->model();
        llama_context * ctx_tgt = llama_init->context();
        if (model == nullptr || ctx_tgt == nullptr) {
            LOG_ERR("%s: failed to init target model/context\n", __func__);
            result = 1;
            goto done;
        }

        auto cparams_mtp = common_context_params_to_llama(params);
        cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cparams_mtp.type_k        = params.speculative.draft.cache_type_k;
        cparams_mtp.type_v        = params.speculative.draft.cache_type_v;
        cparams_mtp.n_rs_seq      = 0;
        cparams_mtp.n_outputs_max = 1;
        cparams_mtp.ctx_other     = ctx_tgt;

        llama_context_ptr ctx_dft(llama_init_from_model(model, cparams_mtp));
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP draft context\n", __func__);
            result = 1;
            goto done;
        }

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();

        common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
        if (spec == nullptr) {
            LOG_ERR("%s: failed to initialize draft-mtp speculation\n", __func__);
            result = 1;
            goto done;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        std::vector<llama_token> committed = common_tokenize(ctx_tgt, params.prompt, true, true);
        if (committed.empty()) {
            LOG_ERR("%s: empty prompt tokenization\n", __func__);
            result = 1;
            goto done;
        }
        const size_t n_prompt = committed.size();

        {
            llama_batch batch = llama_batch_init((int) committed.size(), 0, 1);
            for (int i = 0; i < (int) committed.size(); ++i) {
                common_batch_add(batch, committed[i], i, { 0 }, i == (int) committed.size() - 1);
            }

            if (llama_decode(ctx_tgt, batch) != 0) {
                LOG_ERR("%s: prompt decode failed\n", __func__);
                llama_batch_free(batch);
                result = 1;
                goto done;
            }
            if (!common_speculative_process(spec.get(), batch)) {
                LOG_ERR("%s: MTP prompt process failed\n", __func__);
                llama_batch_free(batch);
                result = 1;
                goto done;
            }
            llama_batch_free(batch);
        }

        common_speculative_begin(spec.get(), 0, committed);

        llama_token sampled = argmax_logits(llama_get_logits_ith(ctx_tgt, -1), n_vocab);
        std::vector<llama_token> generated;
        generated.reserve(params.n_predict + 8);
        generated.push_back(sampled);

        int n_forwards = 0;
        int n_draft_tokens = 0;
        int n_draft_accepted = 0;
        int n_empty_drafts = 0;

        const int64_t t_start_us = ggml_time_us();

        while ((int) generated.size() < params.n_predict && !llama_vocab_is_eog(vocab, sampled)) {
            std::vector<llama_token> draft;
            auto & dp = common_speculative_get_draft_params(spec.get(), 0);
            dp.drafting = true;
            dp.n_max    = params.speculative.draft.n_max;
            dp.n_past   = (llama_pos) committed.size();
            dp.id_last  = sampled;
            dp.prompt   = &committed;
            dp.result   = &draft;

            common_speculative_draft(spec.get());
            n_draft_tokens += (int) draft.size();
            if (draft.empty()) {
                n_empty_drafts++;
            }

            const int n_verify = 1 + (int) draft.size();
            llama_batch batch = llama_batch_init(n_verify, 0, 1);
            llama_pos pos = (llama_pos) committed.size();
            common_batch_add(batch, sampled, pos++, { 0 }, true);
            for (llama_token tok : draft) {
                common_batch_add(batch, tok, pos++, { 0 }, true);
            }

            if (llama_decode(ctx_tgt, batch) != 0) {
                LOG_ERR("%s: verify decode failed at generated=%zu\n", __func__, generated.size());
                llama_batch_free(batch);
                result = 1;
                goto done;
            }
            if (!common_speculative_process(spec.get(), batch)) {
                LOG_ERR("%s: MTP verify process failed at generated=%zu\n", __func__, generated.size());
                llama_batch_free(batch);
                result = 1;
                goto done;
            }

            n_forwards++;

            int n_accept = 0;
            for (; n_accept < (int) draft.size(); ++n_accept) {
                const llama_token truth = argmax_logits(llama_get_logits_ith(ctx_tgt, n_accept), n_vocab);
                if (truth != draft[n_accept]) {
                    break;
                }
            }

            const llama_token next_sampled = argmax_logits(llama_get_logits_ith(ctx_tgt, n_accept), n_vocab);
            llama_batch_free(batch);

            if (!draft.empty()) {
                common_speculative_accept(spec.get(), 0, (uint16_t) n_accept);
            }

            committed.push_back(sampled);
            for (int i = 0; i < n_accept; ++i) {
                committed.push_back(draft[i]);
            }

            if (n_accept < (int) draft.size()) {
                const llama_pos trim_pos = (llama_pos) committed.size();
                if (!context_trim(ctx_tgt, trim_pos) || !context_trim(ctx_dft.get(), trim_pos)) {
                    LOG_ERR("%s: failed to roll back target/draft KV at pos=%d\n", __func__, (int) trim_pos);
                    result = 1;
                    goto done;
                }
            }

            n_draft_accepted += n_accept;
            for (int i = 0; i < n_accept && (int) generated.size() < params.n_predict; ++i) {
                generated.push_back(draft[i]);
            }

            sampled = next_sampled;
            if ((int) generated.size() < params.n_predict) {
                generated.push_back(sampled);
            }
        }

        const int64_t t_end_us = ggml_time_us();
        const double secs = (t_end_us - t_start_us) / 1e6;
        const double alpha = n_draft_tokens > 0 ? (double) n_draft_accepted / (double) n_draft_tokens : 0.0;
        const double tpf = n_forwards > 0 ? (double) generated.size() / (double) n_forwards : 0.0;
        const double tps = secs > 0.0 ? (double) generated.size() / secs : 0.0;

        printf("\n========== TurboQuant MTP probe ==========\n");
        printf("prompt tokens          : %zu\n", n_prompt);
        printf("generated tokens       : %zu\n", generated.size());
        printf("verify forwards        : %d\n", n_forwards);
        printf("draft tokens proposed  : %d\n", n_draft_tokens);
        printf("draft tokens accepted  : %d\n", n_draft_accepted);
        printf("empty draft cycles     : %d\n", n_empty_drafts);
        printf("draft alpha            : %.4f\n", alpha);
        printf("tokens / forward       : %.4f\n", tpf);
        printf("decode wall-clock (s)  : %.4f\n", secs);
        printf("decode tok/s           : %.4f\n", tps);
        printf("==========================================\n");

        common_speculative_print_stats(spec.get());
    }

done:
    llama_backend_free();
    return result;
}
