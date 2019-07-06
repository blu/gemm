//
// openCL-like matrix multiplication - a unit test of compiler's generic-vector capabilities
//

// The next generic-vector capabilities are tested in this 1x4 kernel example
// 1. ability to define generic-vector types via 'vector_size' attribute
// 2. ability to control static-data alignment via 'aligned' attribute
// 3. basic binary arithmetics -- addition and multiplication -- over generic vectors
// 4. assignment-arithmetics ops -- assign-add -- over generic vectors
// 5. array-style vector-lane accessors
//
// Test succeeds and codegen quality matches basic expectations for -DMATX4 on
//   clang 5.0 on armv8
//   g++ 5.4.0 on armv8
//   clang 7.0.0 on armv7 (-mfloat-abi=hard)
//   g++ 7.3.0 on armv7 (-mfloat-abi=hard)
//   clang 5.0.0 on amd64_avx
//   g++ 4.8.1 on amd64_avx (better codegen on 5.1.0)
//   clang 5.0.0 on amd64_sse
//   g++ 4.8.1 on amd64_sse

#include <stdio.h>

typedef __attribute__ ((vector_size(4 * sizeof(float)))) float float4;

__attribute__ ((always_inline)) inline void foo(
	size_t globalIdx,
	size_t globalIdy,
	size_t dim, // dimension in scalars
	const float4 *A,
	const float4 *B,
	float4 *C)
{
	float4 result = (float4){ 0, 0, 0, 0 };
	const size_t dimVec = dim / 4;
	for (size_t i = 0; i < dimVec; ++i) {
		float4 Ai = A[dimVec * globalIdy + i];
		float4 Bi[4];
		Bi[0] = B[dimVec * (i * 4 + 0) + globalIdx];
		Bi[1] = B[dimVec * (i * 4 + 1) + globalIdx];
		Bi[2] = B[dimVec * (i * 4 + 2) + globalIdx];
		Bi[3] = B[dimVec * (i * 4 + 3) + globalIdx];
		result += Ai[0] * Bi[0];
		result += Ai[1] * Bi[1];
		result += Ai[2] * Bi[2];
		result += Ai[3] * Bi[3];
	}

	C[dimVec * globalIdy + globalIdx] = result;
}

#if MATX4
const size_t matxDim = 4;

#else
const size_t matxDim = 8;

#endif
__attribute__ ((aligned(16))) float a[] = {
	 0,  1,  2,  3,  4,  5,  6,  7,
	 8,  9, 10, 11, 12, 13, 14, 15,

#if MATX4 == 0
	16, 17, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39,
	40, 41, 42, 43, 44, 45, 46, 47,
	48, 49, 50, 51, 52, 53, 54, 55,
	56, 57, 58, 59, 60, 61, 62, 63

#endif
};

__attribute__ ((aligned(16))) float b[] = {

#if MATX4
	 2,  0,  0,  0,
	 0,  2,  0,  0,
	 0,  0,  2,  0,
	 0,  0,  0,  2,

#else
	 2,  0,  0,  0,  0,  0,  0,  0,
	 0,  2,  0,  0,  0,  0,  0,  0,
	 0,  0,  2,  0,  0,  0,  0,  0,
	 0,  0,  0,  2,  0,  0,  0,  0,
	 0,  0,  0,  0,  2,  0,  0,  0,
	 0,  0,  0,  0,  0,  2,  0,  0,
	 0,  0,  0,  0,  0,  0,  2,  0,
	 0,  0,  0,  0,  0,  0,  0,  2

#endif
};

__attribute__ ((aligned(16))) float c[matxDim * matxDim];

const size_t vectorDim = sizeof(float4) / sizeof(float);
typedef float4 vector_t;

int main(int, char**) {
	asm volatile ("" ::: "memory"); // lose compiler's tracking of the content of a and b

	for (size_t i = 0; i < matxDim; ++i) {
		for (size_t j = 0; j < matxDim / vectorDim; ++j) {

			foo(j, i, matxDim,
				reinterpret_cast<const vector_t*>(a),
				reinterpret_cast<const vector_t*>(b),
				reinterpret_cast<vector_t*>(c));
		}
	}

	for (size_t i = 0; i < matxDim; ++i) {
		for (size_t j = 0; j < matxDim; ++j) {
			fprintf(stdout, "%f ", c[i * matxDim + j]);
		}
		fputc('\n', stdout);
	}

	return 0;
}
