Musings in GEMM (General Matrix Multiplication)
-----------------------------------------------

Fooling around with flops/clock in the famous SGEMM - what could be more fun? GEMM generally does `C += A * B` where A, B and C are large-ish dense matrices of single (SGEMM) or double precision (DGEMM) floats.

Usage
-----

The low-tech bash script `build_sgemm.sh` will try to build the test for a 64-bit host architecture - substitute the compiler for one of your choice. Macros of interest, passed with `-D` on the command line:

* `ALT` - implementation alternatives (differing by the unrolling of the innermost loop)
	* 0 - scalar version
	* 1 - 16-element-wide version suitable for autovectorizers
	* 2 - 64-element-wide AVX256 version
	* 3 - 128-element-wide AVX256 version
	* 4 - 16-element-wide ASIMD2 (aarch64) version
* `PREFETCH` - distance, in floats, to prefetch in the innermost loop (0 for no prefetch; unused in the scalar version)
* `MATX_SIZE` - dimension of the square matrices A, B & C
* `REP_EXP` - exponent of the number of repetitions of the test, ie. 1eEXP
* `PRINT_MATX` - print out C on the standard output (for debugging)

Tips
----

To tell what prefetch works best on a given CPU and matrix dimension, use something along the following (pick ALT wisely):

	for i in {0..10} ; do ./build_sgemm.sh -DALT=1 -DPREFETCH=`echo "1024 + 512 * $i" | bc` -DMATX_SIZE=512 -DREP_EXP=1 ; ./sgemm ; done

Results
-------

Best results measured in SP flops/clock by the formula:

	(MATX_SIZE^3 * 2) flops_per_matrix * 10^REP_EXP repetitions / (CPU_freq * duration)

| CPU (single thread only)  | width of SIMD ALU | 64x64    | 512x512  | remarks [^1]                                                          |
| ------------------------- | ----------------- | -------- | -------- | --------------------------------------------------------------------- |
| AMD C60 (Bobcat)          | 2-way             | 1.51     | 1.12     | clang++ 3.6, ALT = 1, PREFETCH = 3072, autovectorized SSE2, 800MHz    |
| Intel Core2 T5600         | 4-way             | 3.04     | 2.76     | clang++ 3.4, ALT = 1, PREFETCH = 2560, autovectorized SSE2, 1.83GHz   |
| Intel E5-2687W (SNB)      | 8-way             | 12.86    | 5.46     | clang++ 3.6, ALT = 2, PREFETCH = 2560, AVX256 intrinsics, 3.1GHz      |
| Intel E3-1270v2 (IVB)     | 8-way             | 12.93    | 6.45     | clang++ 3.6, ALT = 2, PREFETCH = 2560, AVX256 intrinsics, 1.6GHz      |
| RK3368 (Cortex-A53)       | 2-way             | 1.84     | 1.12     | clang++ 3.6, ALT = 4, PREFETCH = 2048, ASIMD2 intrinsics, 312MHz [^2] |
| RK3368 (Cortex-A53)       | 2-way             | 1.60     | 0.72     | clang++ 3.6, ALT = 1, PREFETCH = 3584, autovectorized ASIMD2, 1.51GHz |
| MT8163A (Cortex-A53)      | 2-way             | 1.83     | 0.92     | clang++ 3.6, ALT = 4, PREFETCH = 2560, ASIMD2 intrinsics, 1.5GHz      |
| MT8163A (Cortex-A53)      | 2-way             | 2.11     | 0.93     | clang++ 3.6, ALT = 5, PREFETCH = 1536, ASIMD2 intrinsics, 1.5GHz      |

[^1]: Prefetch applies only to 512x512 and is tuned for the given core clock; 64x64 is not prefetched.  
[^2]: Prefetched measurement taken at the lowest CPU clock of 312MHz, to compensate for the low-performance memory bus of the RK3368.
