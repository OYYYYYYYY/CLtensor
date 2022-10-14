#include<stdio.h>
#include<stdlib.h>
#include "TArm.h"

int main(int argc, char *argv[]){
    // 稀疏TTM
    tnsIndex copt_mode = 0;

    // 加载张量tensorx
    tnsSparseTensor tensorx;
    FILE *fp_x = fopen(argv[1], "r");
    tnsLoadSparseTensor(&tensorx, fp_x);
    fclose(fp_x);
    printf("load tensorx\n");

    // 加载稠密矩阵denmat
    tnsDenseMatrix denmat;
    FILE *fp_m = fopen(argv[2], "r");
    tnsLoadDenseMatrix(&denmat, fp_m);
    fclose(fp_m);
    printf("load matrix\n");

    tnsSparseTensor tensory;
    tnsNewSparseTensor(&tensory, tensorx.ndims, tensorx.nmodes, tensorx.nnz);
    tensory.ndims[copt_mode] = denmat.ncols;
    printf("create tensory\n");

    tnsTTMSpatsr(&tensory, &tensorx, &denmat, copt_mode);
    printf("finish compution\n");

    FILE *fp_w = fopen("oydata/spattm.tns", "w");
    tnsDumpSparseTensor(&tensory, fp_w);
    printf("finish output\n");
    
    tnsSparseTensor tensorz;
    tnsNewSparseTensor(&tensorz, tensorx.ndims, tensorx.nmodes, tensorx.nnz);
    tensorz.ndims[copt_mode] = denmat.ncols;
    printf("create tensorz\n");

    tnsOMPTTMSpatsr(&tensorz, &tensorx, &denmat, copt_mode);
    printf("finish compution\n");
    tnsDumpSparseTensor(&tensorz, fp_w);
    printf("finish output\n");

    
    fclose(fp_w);
    tnsFreeSparseTensor(&tensorx);
    tnsFreeSparseTensor(&tensory);
    tnsFreeDenseMatrix(&denmat);

    return 0;
}