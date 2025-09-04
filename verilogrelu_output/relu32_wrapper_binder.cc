#include <cstddef>
#include "binder_util.hh"
#include "Vrelu32_wrapper.h"

struct relu32_wrapper_config {
    static const size_t N_inp = 32;
    static const size_t N_out = 32;
    static const size_t max_inp_bw = 8;
    static const size_t max_out_bw = 7;
    static const size_t II = 0;
    static const size_t latency = 0;
    typedef Vrelu32_wrapper dut_t;
};

extern "C" {
bool openmp_enabled() {
    return _openmp;
}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {
    batch_inference<relu32_wrapper_config>(c_inp, c_out, n_samples);
}
}
