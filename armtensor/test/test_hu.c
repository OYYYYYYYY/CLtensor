#include <TArm.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "omp.h"



int main(void)
{   
    FILE* fp_w = fopen("re.tns", "w");
    //测试张量排序
    tnsSparseTensor tsr;
    FILE* fp_r = fopen("tensor/4d_3_16.tns", "r");
    printf("从文件中读取稀疏张量\n");
    tnsLoadSparseTensor(&tsr, fp_r);
    tnsDumpSparseTensor(&tsr, fp_w);
    tnsSparseTensor tsr_2;
    tnsNewSparseTensor(&tsr_2, tsr.ndims, tsr.nmodes, tsr.nnz);
    tnsIndex order[4]={2,3,0,1};
    tnsPermuteSpatsr(&tsr_2, &tsr, order);
    tnsDumpSparseTensor(&tsr_2, fp_w);

    tnsFreeSparseTensor(&tsr);
    tnsFreeSparseTensor(&tsr_2);
    fclose(fp_r);
    fclose(fp_w);

    return 0;
}



