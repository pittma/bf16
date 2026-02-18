extern int dot_product_bf16_avx512(int a, int b);

int dot_product_asm(int a, int b) {
   return dot_product_bf16_avx512(a, b);
}
