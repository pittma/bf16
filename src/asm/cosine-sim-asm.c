#include "cosine-sim-asm.h"

extern float cosine_sim_bf16_avx512(float *a, float *b, size_t n);

float cosine_sim_asm(float *a, float *b, size_t n) {
   return cosine_sim_bf16_avx512(a, b, n);
}
