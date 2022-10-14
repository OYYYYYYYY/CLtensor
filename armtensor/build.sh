#!/bin/bash

set -e

echo "This script will do an out-of-tree build of TensorBase into the 'build' directory."

cd `dirname $0`

# If you have GCC >= 6.0 and CUDA <= 8.0,
# write this into 'build.config': -DCUDA_HOST_COMPILER=gcc-5
# You can also write other configuation flags into 'build.config'
declare -a CMAKE_FLAGS
[ -e build.config ] && CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" $(<build.config))
CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" "$@")

rm -rf ./build/*
mkdir -p build
cd build

cmake "${CMAKE_FLAGS[@]}" ..

make

echo "Finished. Check the 'build' directory for results."