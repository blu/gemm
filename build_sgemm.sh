#!/bin/bash

CXX=${CXX:-clang++}
CXXFLAGS=(
	-std=c++11
	-O3
	-fstrict-aliasing
	-fno-rtti
	-fno-exceptions
)
CXX_FILENAME=${CXX##*/}
if [[ $CXX_FILENAME == *clang++* ]]; then
	CXXFLAGS+=(
		-ffp-contract=fast
	)

elif [[ $CXX_FILENAME == *g++* ]]; then
	CXXFLAGS+=(
		-ffast-math
	)

elif [[ $CXX_FILENAME == *icpc* ]]; then
	CXXFLAGS+=(
		-fp-model fast=2
		-opt-prefetch=0
		-opt-streaming-cache-evict=0
	)
fi
if [[ $MACHTYPE == *-pc-msys ]]; then
	# nothing mingw-specific, yet
	LFLAGS=()
elif [[ $MACHTYPE == *-apple-darwin* ]]; then
	# nothing darwin-specific, yet
	LFLAGS=()
elif [[ $MACHTYPE == *-linux-* ]]; then
	LFLAGS=(
		-lrt
	)
else
	echo Unknown platform
	exit 255
fi

source cxx_util.sh

# use the machine name to detect armv8 devices with aarch64 kernels and armv7 userspaces
UNAME_MACHINE=`uname -m`

if [[ $UNAME_MACHINE == "aarch64" ]]; then

	cxx_uarch_arm

elif [[ $UNAME_MACHINE == "arm64" ]]; then

	CXXFLAGS+=(
		-DCACHELINE_SIZE=`sysctl hw | grep cachelinesize | sed 's/^hw.cachelinesize: //g'`
		-march=armv8.4-a
		-mtune=native
	)

elif [[ $HOSTTYPE == "i686" ]]; then

	CXXFLAGS+=(
		-DCACHELINE_SIZE=32
		-march=native
		-mtune=native
	)

elif [[ $HOSTTYPE == "x86_64" ]]; then

	CXXFLAGS+=(
		-DCACHELINE_SIZE=64
		-march=native
		-mtune=native
	)

elif [[ $HOSTTYPE == "powerpc64" || $HOSTTYPE == "ppc64" ]]; then

	CXXFLAGS+=(
		-DCACHELINE_SIZE=64
		-mpowerpc64
		-mcpu=powerpc64
		-mtune=power6
		-maltivec
		-mvrsave
	)

elif [[ $HOSTTYPE == "mipsel" ]]; then

	CXXFLAGS+=(
		-DCACHELINE_SIZE=32
		-march=mips32r5
		-mtune=p5600
		-mfp64
		-mhard-float
		-mmadd4
		-mmsa
	)
fi

BUILD_CMD="-o sgemm "${CXXFLAGS[@]}" sgemm.cpp "${LFLAGS[@]}" "${@}
echo $CXX $BUILD_CMD
$CXX $BUILD_CMD
