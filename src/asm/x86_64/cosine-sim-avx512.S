.text

.globl  cosine_sim_bf16_avx512
.type   cosine_sim_bf16_avx512, @function
.align  32
cosine_sim_bf16_avx512:
  .cfi_startproc
  endbr64
  # clear accumulator regs
  vpxord    %xmm8,%xmm8,%xmm8
  vpxord    %xmm9,%xmm9,%xmm9
  vpxord    %xmm10,%xmm10,%xmm10

  _cosine_sim_kernel:
  vmovdqu  (%rdi), %ymm11
  vmovdqu  (%rsi), %ymm12
  vcvtneps2bf16 %ymm11, %xmm11
  vcvtneps2bf16 %ymm12, %xmm12
  # magnitude product
  vdpbf16ps %xmm11, %xmm11, %xmm8
  vdpbf16ps %xmm12, %xmm12, %xmm9
  # dot product
  vdpbf16ps %xmm11, %xmm12, %xmm10

  # loop maintenance
  add   $32, %rdi
  add   $32, %rsi
  sub   $8, %rdx
  jnz   _cosine_sim_kernel

  # magnitude final calc
  vshufps   $0b00001110, %xmm8, %xmm8, %xmm13
  vaddps    %xmm8, %xmm13, %xmm13
  vshufps   $0b00000001, %xmm13, %xmm13, %xmm8
  vaddps    %xmm13, %xmm8, %xmm8
  vshufps   $0b00001110, %xmm9, %xmm9, %xmm14
  vaddps    %xmm9, %xmm14, %xmm14
  vshufps   $0b00000001, %xmm14, %xmm14, %xmm9
  vaddps    %xmm14, %xmm9, %xmm9
  vmulss    %xmm8, %xmm9, %xmm9
  vsqrtss   %xmm9, %xmm9, %xmm9

  # dot prod accum
  vshufps   $0b00001110, %xmm10, %xmm10, %xmm15
  vaddps    %xmm10, %xmm15, %xmm15
  vshufps   $0b00000001, %xmm15, %xmm15, %xmm10
  vaddps    %xmm15, %xmm10, %xmm10

  # final division
  vdivss    %xmm9, %xmm10, %xmm0
  ret
  .cfi_endproc
