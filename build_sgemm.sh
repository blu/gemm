#!/bin/bash

CC=clang++-3.6
CFLAGS=(
	-O3
	-fstrict-aliasing
	-fno-rtti
	-fno-exceptions
)
CC_FILENAME=${CC##*/}
if [[ ${CC_FILENAME:0:3} == "g++" ]]; then
	CFLAGS+=(
		-ffast-math
	)

elif [[ ${CC_FILENAME:0:7} == "clang++" ]]; then
	CFLAGS+=(
		-ffp-contract=fast
	)

elif [[ ${CC_FILENAME:0:4} == "icpc" ]]; then
	CFLAGS+=(
		-fp-model fast=2
		-opt-prefetch=0
		-opt-streaming-cache-evict=0
	)
fi
if [[ ${MACHTYPE} =~ "-apple-darwin" ]]; then
	# nothing darwin-specific, yet
	LFLAGS+=()
elif [[ ${MACHTYPE} =~ "-linux-" ]]; then
	LFLAGS+=(
		-lrt
	)
else
	echo Unknown platform
	exit 255
fi
# use the machine name to detect armv8 devices with aarch64 kernels and armv7 userspaces
UNAME_MACHINE=`uname -m`

if [[ $UNAME_MACHINE == "aarch64" ]]; then

	CFLAGS+=(
		-march=armv8-a
	)

	# clang can fail auto-detecting the host armv8 cpu on some setups; collect all part numbers
	UARCH=`cat /proc/cpuinfo | grep "^CPU part" | sed s/^[^[:digit:]]*//`

	# in order of preference, in case of big.LITTLE
	if   [ `echo $UARCH | grep -c 0xd09` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a73
		)
	elif [ `echo $UARCH | grep -c 0xd08` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a72
		)
	elif [ `echo $UARCH | grep -c 0xd07` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a57
		)
	elif [ `echo $UARCH | grep -c 0xd03` -ne 0 ]; then
		CFLAGS+=(
			-mtune=cortex-a53
		)
	fi

elif [[ $HOSTTYPE == "x86_64" ]]; then

	CFLAGS+=(
		-march=native
		-mtune=native
	)

elif [[ $HOSTTYPE == "powerpc64" || $HOSTTYPE == "ppc64" ]]; then

	CFLAGS+=(
		-mpowerpc64
		-mcpu=powerpc64
		-mtune=power6
		-maltivec
		-mvrsave
	)
fi

BUILD_CMD=${CC}" -o sgemm "${CFLAGS[@]}" sgemm.cpp "${LFLAGS[@]}" "${@}
echo $BUILD_CMD
$BUILD_CMD
