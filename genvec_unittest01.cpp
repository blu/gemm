//
// openCL-like bitonic sorting network - a unit test of compiler's generic-vector capabilities
//

// The next generic-vector capabilities are tested in this bitonic sorting network example
// 1. ability to define generic-vector types via 'vector_size' attribute
// 2. basic boolean arithmetics -- OR and AND -- over generic vectors
// 3. array-style vector-lane accessors
//
// Test succeeds and codegen quality matches basic expectations on
//   clang 9.0 on armv8
//   clang 9.0 on amd64_avx2

#include <stdio.h>

typedef int __attribute__ ((vector_size(sizeof(int) * 4))) vec;
typedef int __attribute__ ((vector_size(sizeof(int) * 8))) vel;

vec min(const vec& a, const vec& b)
{
    const vec mask = a < b;
    return a & mask | b & ~mask;
}

vec max(const vec& a, const vec& b)
{
    const vec mask = a > b;
    return a & mask | b & ~mask;
}

void bitonic_sort(vel &ar)
{
	const vec a0 = (vec){ ar[0], ar[2], ar[4], ar[6] };
	const vec b0 = (vec){ ar[1], ar[3], ar[5], ar[7] };
	const vec min0 = min(a0, b0);
	const vec max0 = max(a0, b0);

	const vec a1 = (vec){ min0[0], min0[1], min0[2], min0[3] };
	const vec b1 = (vec){ max0[1], max0[0], max0[3], max0[2] };
	const vec min1 = min(a1, b1);
	const vec max1 = max(a1, b1);

	const vec a2 = (vec){ min1[0], max1[0], max1[2], min1[2] };
	const vec b2 = (vec){ min1[1], max1[1], max1[3], min1[3] };
	const vec min2 = min(a2, b2);
	const vec max2 = max(a2, b2);

	const vec a3 = (vec){ min2[0], min2[2], min2[1], min2[3] };
	const vec b3 = (vec){ max2[2], max2[0], max2[3], max2[1] };
	const vec min3 = min(a3, b3);
	const vec max3 = max(a3, b3);

	const vec a4 = (vec){ min3[0], min3[1], max3[0], max3[1] };
	const vec b4 = (vec){ min3[2], min3[3], max3[2], max3[3] };
	const vec min4 = min(a4, b4);
	const vec max4 = max(a4, b4);

	const vec a5 = (vec){ min4[0], max4[0], min4[2], max4[2] };
	const vec b5 = (vec){ min4[1], max4[1], min4[3], max4[3] };
	const vec min5 = min(a5, b5);
	const vec max5 = max(a5, b5);

	ar = (vel){ min5[0], max5[0], min5[1], max5[1], min5[2], max5[2], min5[3], max5[3] };
}

vel ar = (vel){ 7, 6, 5, 4, 3, 2, 1, 0 };
vel br = (vel){ 7, 0, 6, 1, 5, 2, 4, 3 };

int main(int, char**)
{
	asm volatile ("" ::: "memory");
	bitonic_sort(ar);

	fprintf(stdout, "%d %d %d %d %d %d %d %d\n",
		ar[0],
		ar[1],
		ar[2],
		ar[3],
		ar[4],
		ar[5],
		ar[6],
		ar[7]);

	bitonic_sort(br);
	fprintf(stdout, "%d %d %d %d %d %d %d %d\n",
		br[0],
		br[1],
		br[2],
		br[3],
		br[4],
		br[5],
		br[6],
		br[7]);

	return 0;
}
