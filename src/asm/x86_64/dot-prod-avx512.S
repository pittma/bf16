.text

# stuff
.globl dot_product_bf16_avx512
.type dot_product_bf16_avx512, @function
.align  32
dot_product_bf16_avx512:
  .cfi_startproc
  endbr64
  add %rdi,%rsi
  mov %rsi,%rax
  ret
  .cfi_endproc
