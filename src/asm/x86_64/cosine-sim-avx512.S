.text

# stuff
.globl  cosine_sim_bf16_avx512
.type   cosine_sim_bf16_avx512, @function
.align  32
cosine_sim_bf16_avx512:
  .cfi_startproc
  endbr64
  vmovdqu  (%rdi), %ymm0
  vmovdqu  (%rsi), %ymm1
  vcvtneps2bf16 %ymm0, %xmm2
  vcvtneps2bf16 %ymm1, %xmm3
  vdpbf16ps %xmm2, %xmm3, %xmm4
  vshufps   $0b00001110, %xmm4, %xmm4, %xmm5
  vaddps    %xmm4, %xmm5, %xmm5
  vshufps   $0b00000001, %xmm5, %xmm5, %xmm4
  vaddps   %xmm5, %xmm4, %xmm4
  ret
  .cfi_endproc
