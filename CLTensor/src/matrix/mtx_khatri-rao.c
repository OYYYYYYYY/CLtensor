#include <TArm.h>
#include <stdlib.h>
#include "omp.h"

static int hadamard_vector(tnsValue *c, tnsValue *a, tnsValue *b, tnsIndex num){
    for(tnsIndex i = 0; i < num; ++i){
        c[i] = a[i] * b[i];
    }
    return 0;
}





/// 稠密矩阵khatrirao 积，矩阵c的空间由外部开辟
int tnsKhatriraoDenmtx(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb){
    tns_CheckError(mtxa->ncols != mtxb->ncols, "tnsKhatrirapDenmtx", "两矩阵的列数不一致");
    tns_CheckError(mtxa->stride != mtxb->stride, "tnsKhatrirapDenmtx", "两矩阵的stride不一致");
    for(tnsIndex i = 0; i < mtxa->nrows; ++i){
        for(tnsIndex j = 0; j < mtxb->nrows; ++j){
            hadamard_vector(mtxc->values.values + (i * mtxb->nrows + j) * mtxb->stride, mtxa->values.values + i * mtxa->stride, mtxb->values.values + j * mtxb->stride, mtxa->ncols);
        }
    }
    return 0;


}
/// 稠密矩阵khatrirao 积，矩阵c的空间由外部开辟，并行方式
int tnsOmpKhatriraoDenmtx(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb, const tnsIndex tk){
    tns_CheckError(mtxa->ncols != mtxb->ncols, "tnsOmpKhatrirapDenmtx", "两矩阵的列数不一致");
    tns_CheckError(mtxa->stride != mtxb->stride, "tnsOmpKhatrirapDenmtx", "两矩阵的stride不一致");
    tnsIndex i;
 #pragma omp parallel for num_threads(tk) private(i) 
    for( i = 0; i < mtxa->nrows; ++i){
        for(tnsIndex j = 0; j < mtxb->nrows; ++j){
            hadamard_vector(mtxc->values.values + (i * mtxb->nrows + j) * mtxb->stride, mtxa->values.values + i * mtxa->stride, mtxb->values.values + j * mtxb->stride, mtxa->ncols);
        }
    }

    return 0;
}




/// 稀疏矩阵khatrirao 积
/*
int tnsKhatriraoSpamtx(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb){
    tns_CheckError(mtxa->ncols!=mtxb->ncols,"tnsKhatrirapSpamtx","两矩阵的列数不一致");
    



}*/

