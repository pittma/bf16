.text

# stuff
.globl  cosine_sim_bf16_avx512
.type   cosine_sim_bf16_avx512, @function
.align  32
cosine_sim_bf16_avx512:
  .cfi_startproc
  endbr64
  vmovdqu  (%rdi), %ymm8
  vmovdqu  (%rsi), %ymm9
  vcvtneps2bf16 %ymm8, %xmm10
  vcvtneps2bf16 %ymm9, %xmm11
  vdpbf16ps %xmm10, %xmm11, %xmm12
  vshufps   $0b00001110, %xmm12, %xmm12, %xmm13
  vaddps    %xmm12, %xmm13, %xmm13
  vshufps   $0b00000001, %xmm13, %xmm13, %xmm12
  vaddps    %xmm13, %xmm12, %xmm0
  ret
  .cfi_endproc
