#include <cstddef>
#include "binder_util.hh"
#include "Vdense0_wrapper.h"

struct dense0_wrapper_config {
    static const size_t N_inp = 64;
    static const size_t N_out = 32;
    static const size_t max_inp_bw = 8;
    static const size_t max_out_bw = 47;
    static const size_t II = 1;
    static const size_t latency = 1;
    typedef Vdense0_wrapper dut_t;
};

extern "C" {
bool openmp_enabled() {
    return _openmp;
}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {
    batch_inference<dense0_wrapper_config>(c_inp, c_out, n_samples);
}
}
