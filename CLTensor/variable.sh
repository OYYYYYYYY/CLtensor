#!/bin/bash

module load compiler/cmake/3.17.2
export HCC_AMDGPU_TARGET=gfx906
export LD_LIBRARY_PATH=/public/software/apps/ghfund/ghfund202107013482/armtensor/build:$LD_LIBRARY_PATH
