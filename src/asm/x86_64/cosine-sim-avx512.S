.macro sum_accum reg_a, reg_b
    vshufps   $0b00001110, \reg_a, \reg_a, \reg_b
    vaddps    \reg_a, \reg_b, \reg_b
    vshufps   $0b00000001, \reg_b, \reg_b, \reg_a
    vaddps    \reg_b, \reg_a, \reg_a
.endm

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
  sum_accum %xmm8, %xmm13
  sum_accum %xmm9, %xmm14
  vmulss    %xmm8, %xmm9, %xmm9
  vsqrtss   %xmm9, %xmm9, %xmm9

  # dot prod accum
  sum_accum %xmm10, %xmm15

  # final division
  vdivss    %xmm9, %xmm10, %xmm0
  ret
  .cfi_endproc
