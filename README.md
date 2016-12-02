Musings in GEMM (General Matrix Multiplication)
-----------------------------------------------

Fooling around with flops/clock in the famous SGEMM - what could be more fun? GEMM generally does `C += A * B` where A, B and C are large-ish dense matrices of single (SGEMM) or double precision (DGEMM) floats.

Usage
-----

The low-tech bash script `build_sgemm.sh` will try to build the test for a 64-bit host architecture - substitute the compiler for one of your choice. Macros of interest, passed with `-D` on the command line:

* ALT
	* 0 - scalar version
	* 1 - 16-element-wide version suitable for autovectorizers
	* 2 - 64-element-wide AVX256 version
	* 3 - 128-element-wide AVX256 version
* PREFETCH - amount of floats to prefetch in the innermost loop (unused in the scalar version)
* MATX_SIZE - dimension of the square matrices A, B & C
* REP_EXP - exponent of the number of repetitions of the test, ie. 1eE
* PRINT_MATX - print out C on the standard output (for debugging)

To tell what prefetch works best on a fiven CPU, use something along the following (pick ALT wisely):

	for i in `seq 0 10`; do ./build_sgemm.sh -DALT=1 -DPREFETCH=`echo "1024 + 512 * $i" | bc` -DPRINT_MATX=0 -DMATX_SIZE=512 -DREP_EXP=1 ; ./sgemm ; done

Best results measured in SP flops/clock by the formula:

	(MATX_SIZE^3 * 2) flops_per_matrix * 10^REP_EXP repetitions / (CPU_freq * duration)

| CPU (single thread only)  | width of SIMD ALU | 64x64    | 512x512  | remarks (prefetch applies only to 512x512)                  |
| ------------------------- | ----------------- | -------- | -------- | ----------------------------------------------------------- |
| AMD C60 (Bobcat)          | 2-way             | 1.51     | 1.12     | clang++ 3.6, ALT = 1, PREFETCH = 3072, autovectorized SSE2  |
| Intel E5-2687W (SNB)      | 8-way             | 12.86    | 5.46     | clang++ 3.6, ALT = 2, PREFETCH = 2560, AVX256 intrinsics    |
| Intel E3-1270v2 (IVB)     | 8-way             | 12.93    | 6.45     | clang++ 3.6, ALT = 2, PREFETCH = 2560, AVX256 intrinsics    |

More to follow.
