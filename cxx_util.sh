#!/bin/bash

#
# update array CXXFLAGS with uarch-specific flags
#
function cxx_uarch_arm() {
	# clang can fail auto-detecting the host armv8 cpu on some setups; collect all part numbers
	VENDOR=`cat /proc/cpuinfo | grep -m 1 "^CPU implementer" | sed s/^[^[:digit:]]*//`
	UARCH=`cat /proc/cpuinfo | grep "^CPU part" | sed s/^[^[:digit:]]*//`

	if   [[ $VENDOR == 0x41 ]]; then # Arm Holdings
		# in order of preference, in case of big.LITTLE
		if   [ `echo $UARCH | grep -c 0xd0c` -ne 0 ]; then # cortex-n1
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a76
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd0b` -ne 0 ]; then # cortex-a76
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a76
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd0a` -ne 0 ]; then # cortex-a75
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a75
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd09` -ne 0 ]; then # cortex-a73
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a73
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd08` -ne 0 ]; then # cortex-a72
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a72
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd07` -ne 0 ]; then # cortex-a57
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a57
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd05` -ne 0 ]; then # cortex-a55
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a55
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd04` -ne 0 ]; then # cortex-a35
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a35
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd03` -ne 0 ]; then # cortex-a53
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a53
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xd01` -ne 0 ]; then # cortex-a32
			CXXFLAGS+=(
				-march=armv8-a
				-mcpu=cortex-a32
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xc0f` -ne 0 ]; then # cortex-a15
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a15
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xc0e` -ne 0 ]; then # cortex-a17
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a17
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xc0d` -ne 0 ]; then # cortex-a12
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a12
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xc09` -ne 0 ]; then # cortex-a9
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a9
				-DCACHELINE_SIZE=32
			)
		elif [ `echo $UARCH | grep -c 0xc08` -ne 0 ]; then # cortex-a8
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a8
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0xc07` -ne 0 ]; then # cortex-a7
			CXXFLAGS+=(
				-march=armv7-a
				-mtune=cortex-a7
				-DCACHELINE_SIZE=32
			)
		else
			echo WARNING: unsupported uarch $UARCH by vendor $VENDOR
			# set compiler flags to something sane
			CXXFLAGS+=(
				-DCACHELINE_SIZE=64
			)
		fi
	elif [[ $VENDOR == 0x46 ]]; then # Fujitsu
		if   [ `echo $UARCH | grep -c 0x001` -ne 0 ]; then # a64fx
			CXXFLAGS+=(
				-march=armv8.2-a+sve
				-mcpu=a64fx
				-DCACHELINE_SIZE=128
			)
		else
			echo WARNING: unsupported uarch $UARCH by vendor $VENDOR
			# set compiler flags to something sane
			CXXFLAGS+=(
				-DCACHELINE_SIZE=128
			)
		fi
	elif [[ $VENDOR == 0x51 ]]; then # Qualcomm
		# in order of preference, in case of big.LITTLE
		if   [ `echo $UARCH | grep -c 0x804` -ne 0 ]; then # kryo 4xx gold
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a76
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0x805` -ne 0 ]; then # kryo 4xx silver
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a55
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0x802` -ne 0 ]; then # kryo 3xx gold
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a75
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0x803` -ne 0 ]; then # kryo 3xx silver
			CXXFLAGS+=(
				-march=armv8.2-a
				-mtune=cortex-a55
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0x800` -ne 0 ]; then # kryo 2xx gold
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a73
				-DCACHELINE_SIZE=64
			)
		elif [ `echo $UARCH | grep -c 0x801` -ne 0 ]; then # kryo 2xx silver
			CXXFLAGS+=(
				-march=armv8-a
				-mtune=cortex-a53
				-DCACHELINE_SIZE=64
			)
		else
			echo WARNING: unsupported uarch $UARCH by vendor $VENDOR
			# set compiler flags to something sane
			CXXFLAGS+=(
				-DCACHELINE_SIZE=64
			)
		fi
	else
		echo WARNING: unsupported uarch vendor $VENDOR
		# set compiler flags to something sane
		CXXFLAGS+=(
			-DCACHELINE_SIZE=64
		)
	fi
}
