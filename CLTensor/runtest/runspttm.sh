#!/bin/bash

tensor=5d_100000000_14m.tns
matrix=matrix_100000000_2.mtx
Nodenum=1
Corenum=1

echo "Tensor: $tensor, matrix: $matrix"
echo "Node number is $Nodenum, core number is $Corenum"
srun -p kshdtest -N ${Nodenum} -n ${Corenum} -c 1 ./build/test/testspttm ./data/${tensor} ./data/${matrix}

echo "Tensor: $tensor, matrix: $matrix"
echo "Dcu number is 4"
srun -p kshdtest -N 1 -n 1 -c 1 --gres=dcu:4 --exclusive ./build/test/testspttm_hip ./data/${tensor} ./data/${matrix}