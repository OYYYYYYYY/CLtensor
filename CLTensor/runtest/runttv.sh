#!/bin/bash

tensor=3d_1024_1100_1024.tns
vector=vector_1024.tns
Nodenum=1
Corenum=1

echo "Tensor: $tensor, vector: $vector"
echo "Node number is $Nodenum, core number is $Corenum"
srun -p kshdtest -N ${Nodenum} -n ${Corenum} -c 1 ./build/test/testdenttv ./data/${tensor} ./data/${vector}

echo "Tensor: $tensor, vector: $vector"
echo "Dcu number is 4"
srun -p kshdtest -N 1 -n 1 -c 1 --gres=dcu:4 --exclusive ./build/test/testdenttv_hip ./data/${tensor} ./data/${vector}