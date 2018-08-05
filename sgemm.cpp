#include <stdio.h>
#include <stdint.h>
#include "timer.h"

#if _LP64 == 1
#define CACHELINE_SIZE 64
#else
#define CACHELINE_SIZE 32
#endif

#define GEMM_PAGE_SIZE 4096

#if !defined(MATX_SIZE)
#define MATX_SIZE 512
#endif

#if !defined(REP_EXP)
#define REP_EXP 1
#endif

#define CATENATE(x, y) x##y
#define CAT(x, y) CATENATE(x, y)

float ma[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(GEMM_PAGE_SIZE)));
float mb[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(GEMM_PAGE_SIZE)));
float mc[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(GEMM_PAGE_SIZE)));

static void fprint_matx(FILE* const out, const float (&mat)[MATX_SIZE][MATX_SIZE]) {
	for (size_t i = 0; i < sizeof(mat) / sizeof(mat[0]); ++i) {
		fprintf(out, "%03lu:", i);
		for (size_t j = 0; j < sizeof(mat[0]) / sizeof(mat[0][0]); ++j)
			fprintf(out, "\t%f", mat[i][j]);
		fprintf(out, "\n");
	}
}

#if PREFETCH != 0
enum {
	prefetch_ro = 0,
	prefetch_rw = 1,
};
enum {
	prefetch_nt = 0,
	prefetch_t1 = 1,
	prefetch_t2 = 2,
	prefetch_t3 = 3
};

#endif

#if ALT == -1
////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm scalar kernel

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t i = 0; i < MATX_SIZE; ++i) {
		for (size_t j = 0; j < MATX_SIZE; ++j) {
			const float ma_ji = ma[j][i];

			for (size_t k = 0; k < MATX_SIZE; ++k)
				mc[j][k] += ma_ji * mb[i][k];
		}
	}
}

#elif ALT == 0
////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 1x16

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

			float mmc[16] = {
				mc[j][k +  0],
				mc[j][k +  1],
				mc[j][k +  2],
				mc[j][k +  3],
				mc[j][k +  4],
				mc[j][k +  5],
				mc[j][k +  6],
				mc[j][k +  7],

				mc[j][k +  8],
				mc[j][k +  9],
				mc[j][k + 10],
				mc[j][k + 11],
				mc[j][k + 12],
				mc[j][k + 13],
				mc[j][k + 14],
				mc[j][k + 15]
			};

			for (size_t i = 0; i < MATX_SIZE; ++i) {

#if PREFETCH != 0
				// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const float mmb[16] = {
					mb[i][k +  0],
					mb[i][k +  1],
					mb[i][k +  2],
					mb[i][k +  3],
					mb[i][k +  4],
					mb[i][k +  5],
					mb[i][k +  6],
					mb[i][k +  7],

					mb[i][k +  8],
					mb[i][k +  9],
					mb[i][k + 10],
					mb[i][k + 11],
					mb[i][k + 12],
					mb[i][k + 13],
					mb[i][k + 14],
					mb[i][k + 15]
				};

				const float ma_ji = ma[j][i];

				mmc[ 0] += ma_ji * mmb[ 0];
				mmc[ 1] += ma_ji * mmb[ 1];
				mmc[ 2] += ma_ji * mmb[ 2];
				mmc[ 3] += ma_ji * mmb[ 3];
				mmc[ 4] += ma_ji * mmb[ 4];
				mmc[ 5] += ma_ji * mmb[ 5];
				mmc[ 6] += ma_ji * mmb[ 6];
				mmc[ 7] += ma_ji * mmb[ 7];

				mmc[ 8] += ma_ji * mmb[ 8];
				mmc[ 9] += ma_ji * mmb[ 9];
				mmc[10] += ma_ji * mmb[10];
				mmc[11] += ma_ji * mmb[11];
				mmc[12] += ma_ji * mmb[12];
				mmc[13] += ma_ji * mmb[13];
				mmc[14] += ma_ji * mmb[14];
				mmc[15] += ma_ji * mmb[15];
			}

			mc[j][k +  0] = mmc[ 0];
			mc[j][k +  1] = mmc[ 1];
			mc[j][k +  2] = mmc[ 2];
			mc[j][k +  3] = mmc[ 3];
			mc[j][k +  4] = mmc[ 4];
			mc[j][k +  5] = mmc[ 5];
			mc[j][k +  6] = mmc[ 6];
			mc[j][k +  7] = mmc[ 7];

			mc[j][k +  8] = mmc[ 8];
			mc[j][k +  9] = mmc[ 9];
			mc[j][k + 10] = mmc[10];
			mc[j][k + 11] = mmc[11];
			mc[j][k + 12] = mmc[12];
			mc[j][k + 13] = mmc[13];
			mc[j][k + 14] = mmc[14];
			mc[j][k + 15] = mmc[15];
		}
	}
}

#elif ALT == 1
#include <xmmintrin.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x16

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; j += 2) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

			__m128 mmc0_0  = _mm_load_ps(&mc[j + 0][k +  0]);
			__m128 mmc0_4  = _mm_load_ps(&mc[j + 0][k +  4]);
			__m128 mmc0_8  = _mm_load_ps(&mc[j + 0][k +  8]);
			__m128 mmc0_12 = _mm_load_ps(&mc[j + 0][k + 12]);

			__m128 mmc1_0  = _mm_load_ps(&mc[j + 1][k +  0]);
			__m128 mmc1_4  = _mm_load_ps(&mc[j + 1][k +  4]);
			__m128 mmc1_8  = _mm_load_ps(&mc[j + 1][k +  8]);
			__m128 mmc1_12 = _mm_load_ps(&mc[j + 1][k + 12]);

			for (size_t i = 0; i < MATX_SIZE; ++i) {

#if PREFETCH != 0
				// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const __m128 mmb0  = _mm_load_ps(&mb[i][k +  0]);
				const __m128 mmb4  = _mm_load_ps(&mb[i][k +  4]);
				const __m128 mmb8  = _mm_load_ps(&mb[i][k +  8]);
				const __m128 mmb12 = _mm_load_ps(&mb[i][k + 12]);

				const __m128 ma0_ji = _mm_load1_ps(&ma[j + 0][i]);
				const __m128 ma1_ji = _mm_load1_ps(&ma[j + 1][i]);

				mmc0_0  += ma0_ji * mmb0;
				mmc0_4  += ma0_ji * mmb4;
				mmc0_8  += ma0_ji * mmb8;
				mmc0_12 += ma0_ji * mmb12;

				mmc1_0  += ma1_ji * mmb0;
				mmc1_4  += ma1_ji * mmb4;
				mmc1_8  += ma1_ji * mmb8;
				mmc1_12 += ma1_ji * mmb12;
			}

			_mm_store_ps(&mc[j + 0][k +  0], mmc0_0);
			_mm_store_ps(&mc[j + 0][k +  4], mmc0_4);
			_mm_store_ps(&mc[j + 0][k +  8], mmc0_8);
			_mm_store_ps(&mc[j + 0][k + 12], mmc0_12);

			_mm_store_ps(&mc[j + 1][k +  0], mmc1_0);
			_mm_store_ps(&mc[j + 1][k +  4], mmc1_4);
			_mm_store_ps(&mc[j + 1][k +  8], mmc1_8);
			_mm_store_ps(&mc[j + 1][k + 12], mmc1_12);
		}
	}
}

#elif ALT == 2
#include <immintrin.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 1x64

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		for (size_t k = 0; k < MATX_SIZE; k += 64) {

			__m256 mmc0  = _mm256_load_ps(&mc[j][k +  0]);
			__m256 mmc8  = _mm256_load_ps(&mc[j][k +  8]);
			__m256 mmc16 = _mm256_load_ps(&mc[j][k + 16]);
			__m256 mmc24 = _mm256_load_ps(&mc[j][k + 24]);
			__m256 mmc32 = _mm256_load_ps(&mc[j][k + 32]);
			__m256 mmc40 = _mm256_load_ps(&mc[j][k + 40]);
			__m256 mmc48 = _mm256_load_ps(&mc[j][k + 48]);
			__m256 mmc56 = _mm256_load_ps(&mc[j][k + 56]);

			for (size_t i = 0; i < MATX_SIZE; ++i) {

#if PREFETCH != 0
				// 64 * sizeof(fp32) = 2^8 bytes = 4 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 2 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 3 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const __m256 mmb0  = _mm256_load_ps(&mb[i][k +  0]);
				const __m256 mmb8  = _mm256_load_ps(&mb[i][k +  8]);
				const __m256 mmb16 = _mm256_load_ps(&mb[i][k + 16]);
				const __m256 mmb24 = _mm256_load_ps(&mb[i][k + 24]);
				const __m256 mmb32 = _mm256_load_ps(&mb[i][k + 32]);
				const __m256 mmb40 = _mm256_load_ps(&mb[i][k + 40]);
				const __m256 mmb48 = _mm256_load_ps(&mb[i][k + 48]);
				const __m256 mmb56 = _mm256_load_ps(&mb[i][k + 56]);

				const __m256 ma_ji = _mm256_broadcast_ss(&ma[j][i]);

				mmc0  += ma_ji * mmb0;
				mmc8  += ma_ji * mmb8;
				mmc16 += ma_ji * mmb16;
				mmc24 += ma_ji * mmb24;
				mmc32 += ma_ji * mmb32;
				mmc40 += ma_ji * mmb40;
				mmc48 += ma_ji * mmb48;
				mmc56 += ma_ji * mmb56;
			}

			_mm256_store_ps(&mc[j][k +  0], mmc0);
			_mm256_store_ps(&mc[j][k +  8], mmc8);
			_mm256_store_ps(&mc[j][k + 16], mmc16);
			_mm256_store_ps(&mc[j][k + 24], mmc24);
			_mm256_store_ps(&mc[j][k + 32], mmc32);
			_mm256_store_ps(&mc[j][k + 40], mmc40);
			_mm256_store_ps(&mc[j][k + 48], mmc48);
			_mm256_store_ps(&mc[j][k + 56], mmc56);
		}
	}
}

#elif ALT == 3
#include <immintrin.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x32

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; j += 2) {
		for (size_t k = 0; k < MATX_SIZE; k += 32) {

			__m256 mmc0_0  = _mm256_load_ps(&mc[j + 0][k +  0]);
			__m256 mmc0_8  = _mm256_load_ps(&mc[j + 0][k +  8]);
			__m256 mmc0_16 = _mm256_load_ps(&mc[j + 0][k + 16]);
			__m256 mmc0_24 = _mm256_load_ps(&mc[j + 0][k + 24]);

			__m256 mmc1_0  = _mm256_load_ps(&mc[j + 1][k +  0]);
			__m256 mmc1_8  = _mm256_load_ps(&mc[j + 1][k +  8]);
			__m256 mmc1_16 = _mm256_load_ps(&mc[j + 1][k + 16]);
			__m256 mmc1_24 = _mm256_load_ps(&mc[j + 1][k + 24]);

			for (size_t i = 0; i < MATX_SIZE; ++i) {

#if PREFETCH != 0
				// 32 * sizeof(fp32) = 2^7 bytes = 2 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const __m256 mmb0  = _mm256_load_ps(&mb[i][k +  0]);
				const __m256 mmb8  = _mm256_load_ps(&mb[i][k +  8]);
				const __m256 mmb16 = _mm256_load_ps(&mb[i][k + 16]);
				const __m256 mmb24 = _mm256_load_ps(&mb[i][k + 24]);

				const __m256 ma0_ji = _mm256_broadcast_ss(&ma[j + 0][i]);
				const __m256 ma1_ji = _mm256_broadcast_ss(&ma[j + 1][i]);

				mmc0_0  += ma0_ji * mmb0;
				mmc0_8  += ma0_ji * mmb8;
				mmc0_16 += ma0_ji * mmb16;
				mmc0_24 += ma0_ji * mmb24;

				mmc1_0  += ma1_ji * mmb0;
				mmc1_8  += ma1_ji * mmb8;
				mmc1_16 += ma1_ji * mmb16;
				mmc1_24 += ma1_ji * mmb24;
			}

			_mm256_store_ps(&mc[j + 0][k +  0], mmc0_0);
			_mm256_store_ps(&mc[j + 0][k +  8], mmc0_8);
			_mm256_store_ps(&mc[j + 0][k + 16], mmc0_16);
			_mm256_store_ps(&mc[j + 0][k + 24], mmc0_24);
                                              
			_mm256_store_ps(&mc[j + 1][k +  0], mmc1_0);
			_mm256_store_ps(&mc[j + 1][k +  8], mmc1_8);
			_mm256_store_ps(&mc[j + 1][k + 16], mmc1_16);
			_mm256_store_ps(&mc[j + 1][k + 24], mmc1_24);
		}
	}
}

#elif ALT == 4
#include <arm_neon.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 1x16

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

			float32x4_t mmc0  = reinterpret_cast< float32x4_t& >(mc[j][k +  0]);
			float32x4_t mmc4  = reinterpret_cast< float32x4_t& >(mc[j][k +  4]);
			float32x4_t mmc8  = reinterpret_cast< float32x4_t& >(mc[j][k +  8]);
			float32x4_t mmc12 = reinterpret_cast< float32x4_t& >(mc[j][k + 12]);

			for (size_t i = 0; i < MATX_SIZE; i += 4) {
				const float32x4_t mma_ji = reinterpret_cast< const float32x4_t& >(ma[j][i]);

#if PREFETCH != 0
				// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const float32x4_t mmb0_0  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  0]);
				const float32x4_t mmb0_4  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  4]);
				const float32x4_t mmb0_8  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  8]);
				const float32x4_t mmb0_12 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 12]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb0_0,  vgetq_lane_f32(mma_ji, 0));
				mmc4  = vmlaq_n_f32(mmc4,  mmb0_4,  vgetq_lane_f32(mma_ji, 0));
				mmc8  = vmlaq_n_f32(mmc8,  mmb0_8,  vgetq_lane_f32(mma_ji, 0));
				mmc12 = vmlaq_n_f32(mmc12, mmb0_12, vgetq_lane_f32(mma_ji, 0));

				const float32x4_t mmb1_0  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  0]);
				const float32x4_t mmb1_4  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  4]);
				const float32x4_t mmb1_8  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  8]);
				const float32x4_t mmb1_12 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 12]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb1_0,  vgetq_lane_f32(mma_ji, 1));
				mmc4  = vmlaq_n_f32(mmc4,  mmb1_4,  vgetq_lane_f32(mma_ji, 1));
				mmc8  = vmlaq_n_f32(mmc8,  mmb1_8,  vgetq_lane_f32(mma_ji, 1));
				mmc12 = vmlaq_n_f32(mmc12, mmb1_12, vgetq_lane_f32(mma_ji, 1));

				const float32x4_t mmb2_0  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  0]);
				const float32x4_t mmb2_4  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  4]);
				const float32x4_t mmb2_8  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  8]);
				const float32x4_t mmb2_12 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 12]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb2_0,  vgetq_lane_f32(mma_ji, 2));
				mmc4  = vmlaq_n_f32(mmc4,  mmb2_4,  vgetq_lane_f32(mma_ji, 2));
				mmc8  = vmlaq_n_f32(mmc8,  mmb2_8,  vgetq_lane_f32(mma_ji, 2));
				mmc12 = vmlaq_n_f32(mmc12, mmb2_12, vgetq_lane_f32(mma_ji, 2));

				const float32x4_t mmb3_0  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  0]);
				const float32x4_t mmb3_4  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  4]);
				const float32x4_t mmb3_8  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  8]);
				const float32x4_t mmb3_12 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 12]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb3_0,  vgetq_lane_f32(mma_ji, 3));
				mmc4  = vmlaq_n_f32(mmc4,  mmb3_4,  vgetq_lane_f32(mma_ji, 3));
				mmc8  = vmlaq_n_f32(mmc8,  mmb3_8,  vgetq_lane_f32(mma_ji, 3));
				mmc12 = vmlaq_n_f32(mmc12, mmb3_12, vgetq_lane_f32(mma_ji, 3));
			}

			reinterpret_cast< float32x4_t& >(mc[j][k +  0]) = mmc0;
			reinterpret_cast< float32x4_t& >(mc[j][k +  4]) = mmc4;
			reinterpret_cast< float32x4_t& >(mc[j][k +  8]) = mmc8;
			reinterpret_cast< float32x4_t& >(mc[j][k + 12]) = mmc12;
		}
	}
}

#elif ALT == 5
#include <arm_neon.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 1x32

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		for (size_t k = 0; k < MATX_SIZE; k += 32) {

			float32x4_t mmc0  = reinterpret_cast< float32x4_t& >(mc[j][k +  0]);
			float32x4_t mmc4  = reinterpret_cast< float32x4_t& >(mc[j][k +  4]);
			float32x4_t mmc8  = reinterpret_cast< float32x4_t& >(mc[j][k +  8]);
			float32x4_t mmc12 = reinterpret_cast< float32x4_t& >(mc[j][k + 12]);

			float32x4_t mmc16 = reinterpret_cast< float32x4_t& >(mc[j][k + 16]);
			float32x4_t mmc20 = reinterpret_cast< float32x4_t& >(mc[j][k + 20]);
			float32x4_t mmc24 = reinterpret_cast< float32x4_t& >(mc[j][k + 24]);
			float32x4_t mmc28 = reinterpret_cast< float32x4_t& >(mc[j][k + 28]);

			for (size_t i = 0; i < MATX_SIZE; i += 4) {
				const float32x4_t mma_ji = reinterpret_cast< const float32x4_t& >(ma[j][i]);

#if PREFETCH != 0
				// 32 * sizeof(fp32) = 2^7 bytes = 2 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const float32x4_t mmb0_0  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  0]);
				const float32x4_t mmb0_4  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  4]);
				const float32x4_t mmb0_8  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  8]);
				const float32x4_t mmb0_12 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 12]);

				const float32x4_t mmb0_16 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 16]);
				const float32x4_t mmb0_20 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 20]);
				const float32x4_t mmb0_24 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 24]);
				const float32x4_t mmb0_28 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 28]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb0_0,  vgetq_lane_f32(mma_ji, 0));
				mmc4  = vmlaq_n_f32(mmc4,  mmb0_4,  vgetq_lane_f32(mma_ji, 0));
				mmc8  = vmlaq_n_f32(mmc8,  mmb0_8,  vgetq_lane_f32(mma_ji, 0));
				mmc12 = vmlaq_n_f32(mmc12, mmb0_12, vgetq_lane_f32(mma_ji, 0));

				mmc16 = vmlaq_n_f32(mmc16, mmb0_16, vgetq_lane_f32(mma_ji, 0));
				mmc20 = vmlaq_n_f32(mmc20, mmb0_20, vgetq_lane_f32(mma_ji, 0));
				mmc24 = vmlaq_n_f32(mmc24, mmb0_24, vgetq_lane_f32(mma_ji, 0));
				mmc28 = vmlaq_n_f32(mmc28, mmb0_28, vgetq_lane_f32(mma_ji, 0));

				const float32x4_t mmb1_0  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  0]);
				const float32x4_t mmb1_4  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  4]);
				const float32x4_t mmb1_8  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  8]);
				const float32x4_t mmb1_12 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 12]);

				const float32x4_t mmb1_16 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 16]);
				const float32x4_t mmb1_20 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 20]);
				const float32x4_t mmb1_24 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 24]);
				const float32x4_t mmb1_28 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 28]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb1_0,  vgetq_lane_f32(mma_ji, 1));
				mmc4  = vmlaq_n_f32(mmc4,  mmb1_4,  vgetq_lane_f32(mma_ji, 1));
				mmc8  = vmlaq_n_f32(mmc8,  mmb1_8,  vgetq_lane_f32(mma_ji, 1));
				mmc12 = vmlaq_n_f32(mmc12, mmb1_12, vgetq_lane_f32(mma_ji, 1));

				mmc16 = vmlaq_n_f32(mmc16, mmb1_16, vgetq_lane_f32(mma_ji, 1));
				mmc20 = vmlaq_n_f32(mmc20, mmb1_20, vgetq_lane_f32(mma_ji, 1));
				mmc24 = vmlaq_n_f32(mmc24, mmb1_24, vgetq_lane_f32(mma_ji, 1));
				mmc28 = vmlaq_n_f32(mmc28, mmb1_28, vgetq_lane_f32(mma_ji, 1));

				const float32x4_t mmb2_0  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  0]);
				const float32x4_t mmb2_4  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  4]);
				const float32x4_t mmb2_8  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  8]);
				const float32x4_t mmb2_12 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 12]);

				const float32x4_t mmb2_16 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 16]);
				const float32x4_t mmb2_20 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 20]);
				const float32x4_t mmb2_24 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 24]);
				const float32x4_t mmb2_28 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 28]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb2_0,  vgetq_lane_f32(mma_ji, 2));
				mmc4  = vmlaq_n_f32(mmc4,  mmb2_4,  vgetq_lane_f32(mma_ji, 2));
				mmc8  = vmlaq_n_f32(mmc8,  mmb2_8,  vgetq_lane_f32(mma_ji, 2));
				mmc12 = vmlaq_n_f32(mmc12, mmb2_12, vgetq_lane_f32(mma_ji, 2));

				mmc16 = vmlaq_n_f32(mmc16, mmb2_16, vgetq_lane_f32(mma_ji, 2));
				mmc20 = vmlaq_n_f32(mmc20, mmb2_20, vgetq_lane_f32(mma_ji, 2));
				mmc24 = vmlaq_n_f32(mmc24, mmb2_24, vgetq_lane_f32(mma_ji, 2));
				mmc28 = vmlaq_n_f32(mmc28, mmb2_28, vgetq_lane_f32(mma_ji, 2));

				const float32x4_t mmb3_0  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  0]);
				const float32x4_t mmb3_4  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  4]);
				const float32x4_t mmb3_8  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  8]);
				const float32x4_t mmb3_12 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 12]);

				const float32x4_t mmb3_16 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 16]);
				const float32x4_t mmb3_20 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 20]);
				const float32x4_t mmb3_24 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 24]);
				const float32x4_t mmb3_28 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 28]);

				mmc0  = vmlaq_n_f32(mmc0,  mmb3_0,  vgetq_lane_f32(mma_ji, 3));
				mmc4  = vmlaq_n_f32(mmc4,  mmb3_4,  vgetq_lane_f32(mma_ji, 3));
				mmc8  = vmlaq_n_f32(mmc8,  mmb3_8,  vgetq_lane_f32(mma_ji, 3));
				mmc12 = vmlaq_n_f32(mmc12, mmb3_12, vgetq_lane_f32(mma_ji, 3));

				mmc16 = vmlaq_n_f32(mmc16, mmb3_16, vgetq_lane_f32(mma_ji, 3));
				mmc20 = vmlaq_n_f32(mmc20, mmb3_20, vgetq_lane_f32(mma_ji, 3));
				mmc24 = vmlaq_n_f32(mmc24, mmb3_24, vgetq_lane_f32(mma_ji, 3));
				mmc28 = vmlaq_n_f32(mmc28, mmb3_28, vgetq_lane_f32(mma_ji, 3));
			}

			reinterpret_cast< float32x4_t& >(mc[j][k +  0]) = mmc0;
			reinterpret_cast< float32x4_t& >(mc[j][k +  4]) = mmc4;
			reinterpret_cast< float32x4_t& >(mc[j][k +  8]) = mmc8;
			reinterpret_cast< float32x4_t& >(mc[j][k + 12]) = mmc12;

			reinterpret_cast< float32x4_t& >(mc[j][k + 16]) = mmc16;
			reinterpret_cast< float32x4_t& >(mc[j][k + 20]) = mmc20;
			reinterpret_cast< float32x4_t& >(mc[j][k + 24]) = mmc24;
			reinterpret_cast< float32x4_t& >(mc[j][k + 28]) = mmc28;
		}
	}
}

#elif ALT == 6
#include <arm_neon.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x16

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; j += 2) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

			float32x4_t mmc0_0  = reinterpret_cast< float32x4_t& >(mc[j + 0][k +  0]);
			float32x4_t mmc0_4  = reinterpret_cast< float32x4_t& >(mc[j + 0][k +  4]);
			float32x4_t mmc0_8  = reinterpret_cast< float32x4_t& >(mc[j + 0][k +  8]);
			float32x4_t mmc0_12 = reinterpret_cast< float32x4_t& >(mc[j + 0][k + 12]);

			float32x4_t mmc1_0  = reinterpret_cast< float32x4_t& >(mc[j + 1][k +  0]);
			float32x4_t mmc1_4  = reinterpret_cast< float32x4_t& >(mc[j + 1][k +  4]);
			float32x4_t mmc1_8  = reinterpret_cast< float32x4_t& >(mc[j + 1][k +  8]);
			float32x4_t mmc1_12 = reinterpret_cast< float32x4_t& >(mc[j + 1][k + 12]);

			for (size_t i = 0; i < MATX_SIZE; i += 4) {
				const float32x4_t mma0_ji = reinterpret_cast< const float32x4_t& >(ma[j + 0][i]);
				const float32x4_t mma1_ji = reinterpret_cast< const float32x4_t& >(ma[j + 1][i]);

#if PREFETCH != 0
				// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const float32x4_t mmb0_0  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  0]);
				const float32x4_t mmb0_4  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  4]);
				const float32x4_t mmb0_8  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  8]);
				const float32x4_t mmb0_12 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 12]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb0_0,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb0_4,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb0_8,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb0_12, vgetq_lane_f32(mma0_ji, 0));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb0_0,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb0_4,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb0_8,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb0_12, vgetq_lane_f32(mma1_ji, 0));

				const float32x4_t mmb1_0  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  0]);
				const float32x4_t mmb1_4  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  4]);
				const float32x4_t mmb1_8  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  8]);
				const float32x4_t mmb1_12 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 12]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb1_0,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb1_4,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb1_8,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb1_12, vgetq_lane_f32(mma0_ji, 1));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb1_0,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb1_4,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb1_8,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb1_12, vgetq_lane_f32(mma1_ji, 1));

				const float32x4_t mmb2_0  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  0]);
				const float32x4_t mmb2_4  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  4]);
				const float32x4_t mmb2_8  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  8]);
				const float32x4_t mmb2_12 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 12]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb2_0,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb2_4,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb2_8,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb2_12, vgetq_lane_f32(mma0_ji, 2));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb2_0,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb2_4,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb2_8,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb2_12, vgetq_lane_f32(mma1_ji, 2));

				const float32x4_t mmb3_0  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  0]);
				const float32x4_t mmb3_4  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  4]);
				const float32x4_t mmb3_8  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  8]);
				const float32x4_t mmb3_12 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 12]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb3_0,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb3_4,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb3_8,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb3_12, vgetq_lane_f32(mma0_ji, 3));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb3_0,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb3_4,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb3_8,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb3_12, vgetq_lane_f32(mma1_ji, 3));
			}

			reinterpret_cast< float32x4_t& >(mc[j + 0][k +  0]) = mmc0_0;
			reinterpret_cast< float32x4_t& >(mc[j + 0][k +  4]) = mmc0_4;
			reinterpret_cast< float32x4_t& >(mc[j + 0][k +  8]) = mmc0_8;
			reinterpret_cast< float32x4_t& >(mc[j + 0][k + 12]) = mmc0_12;

			reinterpret_cast< float32x4_t& >(mc[j + 1][k +  0]) = mmc1_0;
			reinterpret_cast< float32x4_t& >(mc[j + 1][k +  4]) = mmc1_4;
			reinterpret_cast< float32x4_t& >(mc[j + 1][k +  8]) = mmc1_8;
			reinterpret_cast< float32x4_t& >(mc[j + 1][k + 12]) = mmc1_12;
		}
	}
}

#elif ALT == 7
#include <arm_neon.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x32

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	float* mmc0 = mc[0];
	float* mmc1 = mc[1];

	for (size_t j = 0; j < MATX_SIZE; j += 2) {
		for (size_t k = 0; k < MATX_SIZE; k += 32, mmc0 += 32, mmc1 += 32) {

			float32x4_t mmc0_0  = reinterpret_cast< float32x4_t& >(mmc0[ 0]);
			float32x4_t mmc0_4  = reinterpret_cast< float32x4_t& >(mmc0[ 4]);
			float32x4_t mmc0_8  = reinterpret_cast< float32x4_t& >(mmc0[ 8]);
			float32x4_t mmc0_12 = reinterpret_cast< float32x4_t& >(mmc0[12]);

			float32x4_t mmc0_16 = reinterpret_cast< float32x4_t& >(mmc0[16]);
			float32x4_t mmc0_20 = reinterpret_cast< float32x4_t& >(mmc0[20]);
			float32x4_t mmc0_24 = reinterpret_cast< float32x4_t& >(mmc0[24]);
			float32x4_t mmc0_28 = reinterpret_cast< float32x4_t& >(mmc0[28]);

			float32x4_t mmc1_0  = reinterpret_cast< float32x4_t& >(mmc1[ 0]);
			float32x4_t mmc1_4  = reinterpret_cast< float32x4_t& >(mmc1[ 4]);
			float32x4_t mmc1_8  = reinterpret_cast< float32x4_t& >(mmc1[ 8]);
			float32x4_t mmc1_12 = reinterpret_cast< float32x4_t& >(mmc1[12]);

			float32x4_t mmc1_16 = reinterpret_cast< float32x4_t& >(mmc1[16]);
			float32x4_t mmc1_20 = reinterpret_cast< float32x4_t& >(mmc1[20]);
			float32x4_t mmc1_24 = reinterpret_cast< float32x4_t& >(mmc1[24]);
			float32x4_t mmc1_28 = reinterpret_cast< float32x4_t& >(mmc1[28]);

			for (size_t i = 0; i < MATX_SIZE; i += 4) {
				const float32x4_t mma0_ji = reinterpret_cast< const float32x4_t& >(ma[j + 0][i]);
				const float32x4_t mma1_ji = reinterpret_cast< const float32x4_t& >(ma[j + 1][i]);

#if PREFETCH != 0
				// 32 * sizeof(fp32) = 2^7 bytes = 2 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const float32x4_t mmb0_0  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  0]);
				const float32x4_t mmb0_4  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  4]);
				const float32x4_t mmb0_8  = reinterpret_cast< const float32x4_t& >(mb[i + 0][k +  8]);
				const float32x4_t mmb0_12 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 12]);

				const float32x4_t mmb0_16 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 16]);
				const float32x4_t mmb0_20 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 20]);
				const float32x4_t mmb0_24 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 24]);
				const float32x4_t mmb0_28 = reinterpret_cast< const float32x4_t& >(mb[i + 0][k + 28]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb0_0,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb0_4,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb0_8,  vgetq_lane_f32(mma0_ji, 0));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb0_12, vgetq_lane_f32(mma0_ji, 0));

				mmc0_16 = vmlaq_n_f32(mmc0_16, mmb0_16, vgetq_lane_f32(mma0_ji, 0));
				mmc0_20 = vmlaq_n_f32(mmc0_20, mmb0_20, vgetq_lane_f32(mma0_ji, 0));
				mmc0_24 = vmlaq_n_f32(mmc0_24, mmb0_24, vgetq_lane_f32(mma0_ji, 0));
				mmc0_28 = vmlaq_n_f32(mmc0_28, mmb0_28, vgetq_lane_f32(mma0_ji, 0));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb0_0,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb0_4,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb0_8,  vgetq_lane_f32(mma1_ji, 0));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb0_12, vgetq_lane_f32(mma1_ji, 0));

				mmc1_16 = vmlaq_n_f32(mmc1_16, mmb0_16, vgetq_lane_f32(mma1_ji, 0));
				mmc1_20 = vmlaq_n_f32(mmc1_20, mmb0_20, vgetq_lane_f32(mma1_ji, 0));
				mmc1_24 = vmlaq_n_f32(mmc1_24, mmb0_24, vgetq_lane_f32(mma1_ji, 0));
				mmc1_28 = vmlaq_n_f32(mmc1_28, mmb0_28, vgetq_lane_f32(mma1_ji, 0));

				const float32x4_t mmb1_0  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  0]);
				const float32x4_t mmb1_4  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  4]);
				const float32x4_t mmb1_8  = reinterpret_cast< const float32x4_t& >(mb[i + 1][k +  8]);
				const float32x4_t mmb1_12 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 12]);

				const float32x4_t mmb1_16 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 16]);
				const float32x4_t mmb1_20 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 20]);
				const float32x4_t mmb1_24 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 24]);
				const float32x4_t mmb1_28 = reinterpret_cast< const float32x4_t& >(mb[i + 1][k + 28]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb1_0,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb1_4,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb1_8,  vgetq_lane_f32(mma0_ji, 1));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb1_12, vgetq_lane_f32(mma0_ji, 1));

				mmc0_16 = vmlaq_n_f32(mmc0_16, mmb1_16, vgetq_lane_f32(mma0_ji, 1));
				mmc0_20 = vmlaq_n_f32(mmc0_20, mmb1_20, vgetq_lane_f32(mma0_ji, 1));
				mmc0_24 = vmlaq_n_f32(mmc0_24, mmb1_24, vgetq_lane_f32(mma0_ji, 1));
				mmc0_28 = vmlaq_n_f32(mmc0_28, mmb1_28, vgetq_lane_f32(mma0_ji, 1));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb1_0,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb1_4,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb1_8,  vgetq_lane_f32(mma1_ji, 1));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb1_12, vgetq_lane_f32(mma1_ji, 1));

				mmc1_16 = vmlaq_n_f32(mmc1_16, mmb1_16, vgetq_lane_f32(mma1_ji, 1));
				mmc1_20 = vmlaq_n_f32(mmc1_20, mmb1_20, vgetq_lane_f32(mma1_ji, 1));
				mmc1_24 = vmlaq_n_f32(mmc1_24, mmb1_24, vgetq_lane_f32(mma1_ji, 1));
				mmc1_28 = vmlaq_n_f32(mmc1_28, mmb1_28, vgetq_lane_f32(mma1_ji, 1));

				const float32x4_t mmb2_0  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  0]);
				const float32x4_t mmb2_4  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  4]);
				const float32x4_t mmb2_8  = reinterpret_cast< const float32x4_t& >(mb[i + 2][k +  8]);
				const float32x4_t mmb2_12 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 12]);

				const float32x4_t mmb2_16 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 16]);
				const float32x4_t mmb2_20 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 20]);
				const float32x4_t mmb2_24 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 24]);
				const float32x4_t mmb2_28 = reinterpret_cast< const float32x4_t& >(mb[i + 2][k + 28]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb2_0,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb2_4,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb2_8,  vgetq_lane_f32(mma0_ji, 2));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb2_12, vgetq_lane_f32(mma0_ji, 2));

				mmc0_16 = vmlaq_n_f32(mmc0_16, mmb2_16, vgetq_lane_f32(mma0_ji, 2));
				mmc0_20 = vmlaq_n_f32(mmc0_20, mmb2_20, vgetq_lane_f32(mma0_ji, 2));
				mmc0_24 = vmlaq_n_f32(mmc0_24, mmb2_24, vgetq_lane_f32(mma0_ji, 2));
				mmc0_28 = vmlaq_n_f32(mmc0_28, mmb2_28, vgetq_lane_f32(mma0_ji, 2));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb2_0,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb2_4,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb2_8,  vgetq_lane_f32(mma1_ji, 2));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb2_12, vgetq_lane_f32(mma1_ji, 2));

				mmc1_16 = vmlaq_n_f32(mmc1_16, mmb2_16, vgetq_lane_f32(mma1_ji, 2));
				mmc1_20 = vmlaq_n_f32(mmc1_20, mmb2_20, vgetq_lane_f32(mma1_ji, 2));
				mmc1_24 = vmlaq_n_f32(mmc1_24, mmb2_24, vgetq_lane_f32(mma1_ji, 2));
				mmc1_28 = vmlaq_n_f32(mmc1_28, mmb2_28, vgetq_lane_f32(mma1_ji, 2));

				const float32x4_t mmb3_0  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  0]);
				const float32x4_t mmb3_4  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  4]);
				const float32x4_t mmb3_8  = reinterpret_cast< const float32x4_t& >(mb[i + 3][k +  8]);
				const float32x4_t mmb3_12 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 12]);

				const float32x4_t mmb3_16 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 16]);
				const float32x4_t mmb3_20 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 20]);
				const float32x4_t mmb3_24 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 24]);
				const float32x4_t mmb3_28 = reinterpret_cast< const float32x4_t& >(mb[i + 3][k + 28]);

				mmc0_0  = vmlaq_n_f32(mmc0_0,  mmb3_0,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_4  = vmlaq_n_f32(mmc0_4,  mmb3_4,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_8  = vmlaq_n_f32(mmc0_8,  mmb3_8,  vgetq_lane_f32(mma0_ji, 3));
				mmc0_12 = vmlaq_n_f32(mmc0_12, mmb3_12, vgetq_lane_f32(mma0_ji, 3));

				mmc0_16 = vmlaq_n_f32(mmc0_16, mmb3_16, vgetq_lane_f32(mma0_ji, 3));
				mmc0_20 = vmlaq_n_f32(mmc0_20, mmb3_20, vgetq_lane_f32(mma0_ji, 3));
				mmc0_24 = vmlaq_n_f32(mmc0_24, mmb3_24, vgetq_lane_f32(mma0_ji, 3));
				mmc0_28 = vmlaq_n_f32(mmc0_28, mmb3_28, vgetq_lane_f32(mma0_ji, 3));

				mmc1_0  = vmlaq_n_f32(mmc1_0,  mmb3_0,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_4  = vmlaq_n_f32(mmc1_4,  mmb3_4,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_8  = vmlaq_n_f32(mmc1_8,  mmb3_8,  vgetq_lane_f32(mma1_ji, 3));
				mmc1_12 = vmlaq_n_f32(mmc1_12, mmb3_12, vgetq_lane_f32(mma1_ji, 3));

				mmc1_16 = vmlaq_n_f32(mmc1_16, mmb3_16, vgetq_lane_f32(mma1_ji, 3));
				mmc1_20 = vmlaq_n_f32(mmc1_20, mmb3_20, vgetq_lane_f32(mma1_ji, 3));
				mmc1_24 = vmlaq_n_f32(mmc1_24, mmb3_24, vgetq_lane_f32(mma1_ji, 3));
				mmc1_28 = vmlaq_n_f32(mmc1_28, mmb3_28, vgetq_lane_f32(mma1_ji, 3));
			}

			reinterpret_cast< float32x4_t& >(mmc0[ 0]) = mmc0_0;
			reinterpret_cast< float32x4_t& >(mmc0[ 4]) = mmc0_4;
			reinterpret_cast< float32x4_t& >(mmc0[ 8]) = mmc0_8;
			reinterpret_cast< float32x4_t& >(mmc0[12]) = mmc0_12;

			reinterpret_cast< float32x4_t& >(mmc0[16]) = mmc0_16;
			reinterpret_cast< float32x4_t& >(mmc0[20]) = mmc0_20;
			reinterpret_cast< float32x4_t& >(mmc0[24]) = mmc0_24;
			reinterpret_cast< float32x4_t& >(mmc0[28]) = mmc0_28;

			reinterpret_cast< float32x4_t& >(mmc1[ 0]) = mmc1_0;
			reinterpret_cast< float32x4_t& >(mmc1[ 4]) = mmc1_4;
			reinterpret_cast< float32x4_t& >(mmc1[ 8]) = mmc1_8;
			reinterpret_cast< float32x4_t& >(mmc1[12]) = mmc1_12;

			reinterpret_cast< float32x4_t& >(mmc1[16]) = mmc1_16;
			reinterpret_cast< float32x4_t& >(mmc1[20]) = mmc1_20;
			reinterpret_cast< float32x4_t& >(mmc1[24]) = mmc1_24;
			reinterpret_cast< float32x4_t& >(mmc1[28]) = mmc1_28;
		}
	}
}

#elif ALT == 8
#include <msa.h>

////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm kernel window of 2x16

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; j += 2) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

			v4f32 mmc0_0  = reinterpret_cast< const v4f32& >(mc[j + 0][k +  0]);
			v4f32 mmc0_4  = reinterpret_cast< const v4f32& >(mc[j + 0][k +  4]);
			v4f32 mmc0_8  = reinterpret_cast< const v4f32& >(mc[j + 0][k +  8]);
			v4f32 mmc0_12 = reinterpret_cast< const v4f32& >(mc[j + 0][k + 12]);

			v4f32 mmc1_0  = reinterpret_cast< const v4f32& >(mc[j + 1][k +  0]);
			v4f32 mmc1_4  = reinterpret_cast< const v4f32& >(mc[j + 1][k +  4]);
			v4f32 mmc1_8  = reinterpret_cast< const v4f32& >(mc[j + 1][k +  8]);
			v4f32 mmc1_12 = reinterpret_cast< const v4f32& >(mc[j + 1][k + 12]);

			for (size_t i = 0; i < MATX_SIZE; i += 4) { // unroll by 4 to utilize splat instructions -- we have enough regs to afford it
				const v4f32 ma0_i = reinterpret_cast< const v4f32& >(ma[j + 0][i]);
				const v4f32 ma1_i = reinterpret_cast< const v4f32& >(ma[j + 1][i]);

#if PREFETCH != 0
				// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines = 2 * 32-byte cachelines (TODO cacheline-aware prefetch helper)
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 0][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 1][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 2][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i + 3][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const v4f32 mmbi0_0  = reinterpret_cast< const v4f32& >(mb[i + 0][k +  0]);
				const v4f32 mmbi0_4  = reinterpret_cast< const v4f32& >(mb[i + 0][k +  4]);
				const v4f32 mmbi0_8  = reinterpret_cast< const v4f32& >(mb[i + 0][k +  8]);
				const v4f32 mmbi0_12 = reinterpret_cast< const v4f32& >(mb[i + 0][k + 12]);

				const v4f32 mmbi1_0  = reinterpret_cast< const v4f32& >(mb[i + 1][k +  0]);
				const v4f32 mmbi1_4  = reinterpret_cast< const v4f32& >(mb[i + 1][k +  4]);
				const v4f32 mmbi1_8  = reinterpret_cast< const v4f32& >(mb[i + 1][k +  8]);
				const v4f32 mmbi1_12 = reinterpret_cast< const v4f32& >(mb[i + 1][k + 12]);

				const v4f32 mmbi2_0  = reinterpret_cast< const v4f32& >(mb[i + 2][k +  0]);
				const v4f32 mmbi2_4  = reinterpret_cast< const v4f32& >(mb[i + 2][k +  4]);
				const v4f32 mmbi2_8  = reinterpret_cast< const v4f32& >(mb[i + 2][k +  8]);
				const v4f32 mmbi2_12 = reinterpret_cast< const v4f32& >(mb[i + 2][k + 12]);

				const v4f32 mmbi3_0  = reinterpret_cast< const v4f32& >(mb[i + 3][k +  0]);
				const v4f32 mmbi3_4  = reinterpret_cast< const v4f32& >(mb[i + 3][k +  4]);
				const v4f32 mmbi3_8  = reinterpret_cast< const v4f32& >(mb[i + 3][k +  8]);
				const v4f32 mmbi3_12 = reinterpret_cast< const v4f32& >(mb[i + 3][k + 12]);

				const v4i32& ima0_ji0 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma0_i), 0);
				const v4i32& ima1_ji0 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma1_i), 0);
				const v4f32 ma0_ji0 = reinterpret_cast< const v4f32& >(ima0_ji0);
				const v4f32 ma1_ji0 = reinterpret_cast< const v4f32& >(ima1_ji0);

				mmc0_0  += ma0_ji0 * mmbi0_0;
				mmc0_4  += ma0_ji0 * mmbi0_4;
				mmc0_8  += ma0_ji0 * mmbi0_8;
				mmc0_12 += ma0_ji0 * mmbi0_12;

				mmc1_0  += ma1_ji0 * mmbi0_0;
				mmc1_4  += ma1_ji0 * mmbi0_4;
				mmc1_8  += ma1_ji0 * mmbi0_8;
				mmc1_12 += ma1_ji0 * mmbi0_12;

				const v4i32& ima0_ji1 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma0_i), 1);
				const v4i32& ima1_ji1 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma1_i), 1);
				const v4f32 ma0_ji1 = reinterpret_cast< const v4f32& >(ima0_ji1);
				const v4f32 ma1_ji1 = reinterpret_cast< const v4f32& >(ima1_ji1);

				mmc0_0  += ma0_ji1 * mmbi1_0;
				mmc0_4  += ma0_ji1 * mmbi1_4;
				mmc0_8  += ma0_ji1 * mmbi1_8;
				mmc0_12 += ma0_ji1 * mmbi1_12;

				mmc1_0  += ma1_ji1 * mmbi1_0;
				mmc1_4  += ma1_ji1 * mmbi1_4;
				mmc1_8  += ma1_ji1 * mmbi1_8;
				mmc1_12 += ma1_ji1 * mmbi1_12;

				const v4i32& ima0_ji2 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma0_i), 2);
				const v4i32& ima1_ji2 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma1_i), 2);
				const v4f32 ma0_ji2 = reinterpret_cast< const v4f32& >(ima0_ji2);
				const v4f32 ma1_ji2 = reinterpret_cast< const v4f32& >(ima1_ji2);

				mmc0_0  += ma0_ji2 * mmbi2_0;
				mmc0_4  += ma0_ji2 * mmbi2_4;
				mmc0_8  += ma0_ji2 * mmbi2_8;
				mmc0_12 += ma0_ji2 * mmbi2_12;

				mmc1_0  += ma1_ji2 * mmbi2_0;
				mmc1_4  += ma1_ji2 * mmbi2_4;
				mmc1_8  += ma1_ji2 * mmbi2_8;
				mmc1_12 += ma1_ji2 * mmbi2_12;

				const v4i32& ima0_ji3 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma0_i), 3);
				const v4i32& ima1_ji3 = __msa_splati_w(reinterpret_cast< const v4i32& >(ma1_i), 3);
				const v4f32 ma0_ji3 = reinterpret_cast< const v4f32& >(ima0_ji3);
				const v4f32 ma1_ji3 = reinterpret_cast< const v4f32& >(ima1_ji3);

				mmc0_0  += ma0_ji3 * mmbi3_0;
				mmc0_4  += ma0_ji3 * mmbi3_4;
				mmc0_8  += ma0_ji3 * mmbi3_8;
				mmc0_12 += ma0_ji3 * mmbi3_12;

				mmc1_0  += ma1_ji3 * mmbi3_0;
				mmc1_4  += ma1_ji3 * mmbi3_4;
				mmc1_8  += ma1_ji3 * mmbi3_8;
				mmc1_12 += ma1_ji3 * mmbi3_12;
			}

			reinterpret_cast< v4f32& >(mc[j + 0][k +  0]) = mmc0_0;
			reinterpret_cast< v4f32& >(mc[j + 0][k +  4]) = mmc0_4;
			reinterpret_cast< v4f32& >(mc[j + 0][k +  8]) = mmc0_8;
			reinterpret_cast< v4f32& >(mc[j + 0][k + 12]) = mmc0_12;

			reinterpret_cast< v4f32& >(mc[j + 1][k +  0]) = mmc1_0;
			reinterpret_cast< v4f32& >(mc[j + 1][k +  4]) = mmc1_4;
			reinterpret_cast< v4f32& >(mc[j + 1][k +  8]) = mmc1_8;
			reinterpret_cast< v4f32& >(mc[j + 1][k + 12]) = mmc1_12;
		}
	}
}

#else
	#error unknown ALT

#endif
int main(int, char**) {
	for (size_t i = 0; i < MATX_SIZE; ++i) {
		ma[i][i] = 2.f;
		for (size_t j = 0; j < MATX_SIZE; ++j)
			mb[i][j] = i * MATX_SIZE + j;
	}

	const size_t rep = size_t(CAT(1e, REP_EXP));
	const uint64_t t0 = timer_ns();

	for (size_t r = 0; r < rep; ++r) {
		asm volatile ("" : : : "memory");
		matmul(ma, mb, mc);
	}

	const uint64_t dt = timer_ns() - t0;

#if PRINT_MATX != 0
	fprint_matx(stdout, mc);

#endif
	fprintf(stdout, "%f\n", 1e-9 * dt);
	return 0;
}
