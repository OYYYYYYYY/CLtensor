#include<stdio.h>
#include<stdlib.h>
#include "omp.h"
#include "TArm.h"

int main(void){
    // test tensor matricization

    tnsIndex cutpoint = 2;   ///< 设置断点
    tnsSparseTensor spatsra;
    FILE *fp_spa = fopen("tensor/4d_3_16.tns", "r");
    tnsLoadSparseTensor(&spatsra, fp_spa);
    fclose(fp_spa);

    tnsIndex row = 1;
    tnsIndex col = 1;
    for(tnsIndex i = 0; i < cutpoint; ++i)
        row *= spatsra.ndims[i];   ///< 矩阵化后的行大小
    for(tnsIndex i = cutpoint; i < spatsra.nmodes; ++i)
        col *= spatsra.ndims[i];   ///< 矩阵化后的行大小
    
    FILE *fp_w = fopen("oydata/matricization.mtx", "w");

    tnsSparseMatrix spamtxa;
    tnsNewSparseMatrix(&spamtxa, row, col);

    tnsSparseMatrix spamtxb;
    tnsNewSparseMatrix(&spamtxb, row, col);

    tnsMatricizationSpatsr(&spamtxa, &spatsra, cutpoint);
    tnsDumpSparseMatrix(&spamtxa, fp_w);

    tnsOmpMatricizationSpatsr(&spamtxb, &spatsra, cutpoint);
    tnsDumpSparseMatrix(&spamtxb, fp_w);
    
    fclose(fp_w);

    tnsFreeSparseTensor(&spatsra);
    tnsFreeSparseMatrix(&spamtxa);

    return 0;
}