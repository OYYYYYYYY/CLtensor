#!/bin/bash

set -e

echo "This script will do an out-of-tree build of TensorBase into the 'build' directory."

export LD_LIBRARY_PATH=/public/home/ghfund_1_a6/armtensor/build:$LD_LIBRARY_PATH


cd `dirname $0`

# If you have GCC >= 6.0 and CUDA <= 8.0,
# write this into 'build.config': -DCUDA_HOST_COMPILER=gcc-5
# You can also write other configuation flags into 'build.config'
declare -a CMAKE_FLAGS
[ -e build.config ] && CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" $(<build.config))
CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" "$@")

cd ./MTTKRP
g++ -c -o util.o util.cpp -std=c++11 -O3 -g  -w
hipcc -c -o mttkrp.o mttkrp.cu -O3 -w --std=c++11 -Xcompiler
hipcc -c -o mttkrp_gpu.o mttkrp_gpu.cu -O3 -w --std=c++11 -Xcompiler 
hipcc -o mttkrp util.o mttkrp_gpu.o mttkrp.o   -O3 -w  --std=c++11 -Xcompiler 
cd ..

#rm -rf ./build/*
mkdir -p build
cd build

cmake "${CMAKE_FLAGS[@]}" ..

make

echo "Finished. Check the 'build' directory for results."
