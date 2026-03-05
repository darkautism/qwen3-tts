#include "tts_transformer.h"

#include <cstdint>
#include <new>
#include <vector>

struct code_pred_handle {
    qwen3_tts::TTSTransformer * model;
};

extern "C" {

void * code_pred_load(const char * gguf_path, int n_threads) {
    (void)n_threads;
    if (!gguf_path) {
        return nullptr;
    }

    auto * handle = new (std::nothrow) code_pred_handle{nullptr};
    if (!handle) {
        return nullptr;
    }

    handle->model = new (std::nothrow) qwen3_tts::TTSTransformer();
    if (!handle->model) {
        delete handle;
        return nullptr;
    }

    if (!handle->model->load_model(gguf_path)) {
        delete handle->model;
        delete handle;
        return nullptr;
    }

    return handle;
}

int code_pred_predict(void * opaque, const float * hidden, int codebook_0_token, int * output,
                      float temperature, int top_k) {
    if (!opaque || !hidden || !output) {
        return -1;
    }

    auto * handle = static_cast<code_pred_handle *>(opaque);
    std::vector<int32_t> codes;
    if (!handle->model->predict_codes_autoregressive(hidden, codebook_0_token, codes, temperature,
                                                     top_k)) {
        return -2;
    }

    if (codes.size() < 15) {
        return -3;
    }

    for (int i = 0; i < 15; ++i) {
        output[i] = codes[i];
    }

    return 0;
}

void code_pred_free(void * opaque) {
    auto * handle = static_cast<code_pred_handle *>(opaque);
    if (!handle) {
        return;
    }
    if (handle->model) {
        handle->model->unload_model();
        delete handle->model;
    }
    delete handle;
}

} // extern "C"
