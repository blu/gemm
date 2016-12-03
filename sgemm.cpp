#include "stdio.h"
#include "stdint.h"
#include "timer.h"

#define CACHELINE_SIZE 64

#if !defined(MATX_SIZE)
#define MATX_SIZE 512
#endif

#if !defined(REP_EXP)
#define REP_EXP 1
#endif

#define CATENATE(x, y) x##y
#define CAT(x, y) CATENATE(x, y)

float ma[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(CACHELINE_SIZE)));
float mb[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(CACHELINE_SIZE)));
float mc[MATX_SIZE][MATX_SIZE] __attribute__ ((aligned(CACHELINE_SIZE)));

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

#if ALT == 0
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

#elif ALT == 1
static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		for (size_t k = 0; k < MATX_SIZE; k += 16) {

#if PREFETCH != 0
			// 16 * sizeof(fp32) = 2^6 bytes = 1 * 64-byte cachelines
			__builtin_prefetch(((const int8_t*) &mc[j][k + PREFETCH / 2]) + 0 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);

#endif
			float mmc[16] __attribute__ ((aligned(32))) = {
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
				const float mmb[16] __attribute__ ((aligned(32))) = {
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

#elif ALT == 2
#include <immintrin.h>

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		float* mcp = &mc[j][0];
		for (size_t k = 0; k < MATX_SIZE; k += 64) {

#if PREFETCH != 0
			// 64 * sizeof(fp32) = 2^8 bytes = 4 * 64-byte cachelines
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 0 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 1 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 2 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 3 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);

#endif
			__m256 mmc0  = _mm256_load_ps(mcp +  0);
			__m256 mmc8  = _mm256_load_ps(mcp +  8);
			__m256 mmc16 = _mm256_load_ps(mcp + 16);
			__m256 mmc24 = _mm256_load_ps(mcp + 24);
			__m256 mmc32 = _mm256_load_ps(mcp + 32);
			__m256 mmc40 = _mm256_load_ps(mcp + 40);
			__m256 mmc48 = _mm256_load_ps(mcp + 48);
			__m256 mmc56 = _mm256_load_ps(mcp + 56);

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

#if MADD != 0
				mmc0  += ma_ji * mmb0;
				mmc8  += ma_ji * mmb8;
				mmc16 += ma_ji * mmb16;
				mmc24 += ma_ji * mmb24;

				mmc32 += ma_ji * mmb32;
				mmc40 += ma_ji * mmb40;
				mmc48 += ma_ji * mmb48;
				mmc56 += ma_ji * mmb56;

#else
				const __m256 p0  = ma_ji * mmb0;
				const __m256 p8  = ma_ji * mmb8;
				const __m256 p16 = ma_ji * mmb16;
				const __m256 p24 = ma_ji * mmb24;

				mmc0  += p0;
				mmc8  += p8;
				mmc16 += p16;
				mmc24 += p24;

				const __m256 p32 = ma_ji * mmb32;
				const __m256 p40 = ma_ji * mmb40;
				const __m256 p48 = ma_ji * mmb48;
				const __m256 p56 = ma_ji * mmb56;

				mmc32 += p32;
				mmc40 += p40;
				mmc48 += p48;
				mmc56 += p56;

#endif
			}

			_mm256_store_ps(mcp +  0, mmc0);
			_mm256_store_ps(mcp +  8, mmc8);
			_mm256_store_ps(mcp + 16, mmc16);
			_mm256_store_ps(mcp + 24, mmc24);
			_mm256_store_ps(mcp + 32, mmc32);
			_mm256_store_ps(mcp + 40, mmc40);
			_mm256_store_ps(mcp + 48, mmc48);
			_mm256_store_ps(mcp + 56, mmc56);

			mcp += 64;
		}
	}
}

#elif ALT == 3
#include <immintrin.h>

static void matmul(
	const float (&ma)[MATX_SIZE][MATX_SIZE],
	const float (&mb)[MATX_SIZE][MATX_SIZE],
	float (&mc)[MATX_SIZE][MATX_SIZE]) {

	for (size_t j = 0; j < MATX_SIZE; ++j) {
		float* mcp = &mc[j][0];
		for (size_t k = 0; k < MATX_SIZE; k += 128) {

#if PREFETCH != 0
			// 128 * sizeof(fp32) = 2^9 bytes = 8 * 64-byte cachelines
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 0 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 1 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 2 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 3 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);

			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 4 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 5 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 6 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);
			__builtin_prefetch(((const int8_t*) (mcp + PREFETCH / 2)) + 7 * CACHELINE_SIZE, prefetch_rw, prefetch_t3);

#endif
			__m256 mmc0   = _mm256_load_ps(mcp +   0);
			__m256 mmc8   = _mm256_load_ps(mcp +   8);
			__m256 mmc16  = _mm256_load_ps(mcp +  16);
			__m256 mmc24  = _mm256_load_ps(mcp +  24);
			__m256 mmc32  = _mm256_load_ps(mcp +  32);
			__m256 mmc40  = _mm256_load_ps(mcp +  40);
			__m256 mmc48  = _mm256_load_ps(mcp +  48);
			__m256 mmc56  = _mm256_load_ps(mcp +  56);

			__m256 mmc64  = _mm256_load_ps(mcp +  64);
			__m256 mmc72  = _mm256_load_ps(mcp +  72);
			__m256 mmc80  = _mm256_load_ps(mcp +  80);
			__m256 mmc88  = _mm256_load_ps(mcp +  88);
			__m256 mmc96  = _mm256_load_ps(mcp +  96);
			__m256 mmc104 = _mm256_load_ps(mcp + 104);
			__m256 mmc112 = _mm256_load_ps(mcp + 112);
			__m256 mmc120 = _mm256_load_ps(mcp + 120);

			for (size_t i = 0; i < MATX_SIZE; ++i) {

#if PREFETCH != 0
				// 128 * sizeof(fp32) = 2^9 bytes = 8 * 64-byte cachelines
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 0 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 1 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 2 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 3 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const __m256 mmb0   = _mm256_load_ps(&mb[i][k +   0]);
				const __m256 mmb8   = _mm256_load_ps(&mb[i][k +   8]);
				const __m256 mmb16  = _mm256_load_ps(&mb[i][k +  16]);
				const __m256 mmb24  = _mm256_load_ps(&mb[i][k +  24]);
				const __m256 mmb32  = _mm256_load_ps(&mb[i][k +  32]);
				const __m256 mmb40  = _mm256_load_ps(&mb[i][k +  40]);
				const __m256 mmb48  = _mm256_load_ps(&mb[i][k +  48]);
				const __m256 mmb56  = _mm256_load_ps(&mb[i][k +  56]);

#if PREFETCH != 0
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 4 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 5 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 6 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);
				__builtin_prefetch(((const int8_t*) &mb[i][k + PREFETCH]) + 7 * CACHELINE_SIZE, prefetch_ro, prefetch_t3);

#endif
				const __m256 mmb64  = _mm256_load_ps(&mb[i][k +  64]);
				const __m256 mmb72  = _mm256_load_ps(&mb[i][k +  72]);
				const __m256 mmb80  = _mm256_load_ps(&mb[i][k +  80]);
				const __m256 mmb88  = _mm256_load_ps(&mb[i][k +  88]);
				const __m256 mmb96  = _mm256_load_ps(&mb[i][k +  96]);
				const __m256 mmb104 = _mm256_load_ps(&mb[i][k + 104]);
				const __m256 mmb112 = _mm256_load_ps(&mb[i][k + 112]);
				const __m256 mmb120 = _mm256_load_ps(&mb[i][k + 120]);

				const __m256 ma_ji = _mm256_broadcast_ss(&ma[j][i]);

				mmc0   += ma_ji * mmb0;
				mmc8   += ma_ji * mmb8;
				mmc16  += ma_ji * mmb16;
				mmc24  += ma_ji * mmb24;
				mmc32  += ma_ji * mmb32;
				mmc40  += ma_ji * mmb40;
				mmc48  += ma_ji * mmb48;
				mmc56  += ma_ji * mmb56;

				mmc64  += ma_ji * mmb64;
				mmc72  += ma_ji * mmb72;
				mmc80  += ma_ji * mmb80;
				mmc88  += ma_ji * mmb88;
				mmc96  += ma_ji * mmb96;
				mmc104 += ma_ji * mmb104;
				mmc112 += ma_ji * mmb112;
				mmc120 += ma_ji * mmb120;
			}

			_mm256_store_ps(mcp +   0, mmc0);
			_mm256_store_ps(mcp +   8, mmc8);
			_mm256_store_ps(mcp +  16, mmc16);
			_mm256_store_ps(mcp +  24, mmc24);
			_mm256_store_ps(mcp +  32, mmc32);
			_mm256_store_ps(mcp +  40, mmc40);
			_mm256_store_ps(mcp +  48, mmc48);
			_mm256_store_ps(mcp +  56, mmc56);

			_mm256_store_ps(mcp +  64, mmc64);
			_mm256_store_ps(mcp +  72, mmc72);
			_mm256_store_ps(mcp +  80, mmc80);
			_mm256_store_ps(mcp +  88, mmc88);
			_mm256_store_ps(mcp +  96, mmc96);
			_mm256_store_ps(mcp + 104, mmc104);
			_mm256_store_ps(mcp + 112, mmc112);
			_mm256_store_ps(mcp + 120, mmc120);

			mcp += 128;
		}
	}
}

#endif
int main(int, char**) {
	for (size_t i = 0; i < MATX_SIZE; ++i) {
		ma[i][i] = 2.f;
		for (size_t j = 0; j < MATX_SIZE; ++j)
			mb[i][j] = i * MATX_SIZE + j;
	}

	const size_t rep = size_t(CAT(1e, REP_EXP));
	const uint64_t t0 = timer_nsec();

	for (size_t r = 0; r < rep; ++r) {
		asm volatile ("" : : : "memory");
		matmul(ma, mb, mc);
	}

	const uint64_t dt = timer_nsec() - t0;

#if PRINT_MATX != 0
	fprint_matx(stdout, mc);

#endif
	fprintf(stdout, "%f\n", 1e-9 * dt);
	return 0;
}