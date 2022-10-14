#include "omp.h"
#include <TArm.h>
#include <stdlib.h>

#define NTHREADS 8

/*
*  稀疏矩阵部分
*/
/// 稀疏矩阵Hadamard积
int tnsHadamardSpamat(tnsSparseMatrix *mtxc, const tnsSparseMatrix *mtxa, const tnsSparseMatrix *mtxb) { 
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    
    tnsIndex maxnnz = (mtxa->nnz > mtxb->nnz) ? mtxa->nnz : mtxb->nnz;
    mtxc->rowinds.nlens = maxnnz;
    mtxc->colinds.nlens = maxnnz;
    mtxc->values.nlens = maxnnz;
    mtxc->rowinds.values = realloc(mtxc->rowinds.values, (maxnnz) * sizeof mtxc->rowinds);
    mtxc->colinds.values = realloc(mtxc->colinds.values, (maxnnz) * sizeof mtxc->colinds);
    mtxc->values.values = realloc(mtxc->values.values, (maxnnz) * sizeof mtxc->values);
    
    for(tnsIndex i = 0; i < mtxa->nnz; ++i){
        for(tnsIndex j = 0; j < mtxb->nnz; ++j){
            if(mtxa->rowinds.values[i] == mtxb->rowinds.values[j] && mtxa->colinds.values[i] == mtxb->colinds.values[j]){
                mtxc->rowinds.values[mtxc->nnz] = mtxa->rowinds.values[i];
                mtxc->colinds.values[mtxc->nnz] = mtxa->colinds.values[i];
                mtxc->values.values[mtxc->nnz] = mtxa->values.values[i] * mtxb->values.values[j];
                ++mtxc->nnz;
            }///< if
        }///< for j
    }///< for i
    
    mtxc->rowinds.nlens = mtxc->nnz;
    mtxc->colinds.nlens = mtxc->nnz;
    mtxc->values.nlens = mtxc->nnz;
    return 0;
}


/// 并行稀疏矩阵Hadamard积
int tnsOmpHadamardSpamat(tnsSparseMatrix *mtxc, const tnsSparseMatrix *mtxa, const tnsSparseMatrix *mtxb) { 
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }

    tnsIndex maxnnz = (mtxa->nnz > mtxb->nnz) ? mtxa->nnz : mtxb->nnz;
    mtxc->rowinds.nlens = maxnnz;
    mtxc->colinds.nlens = maxnnz;
    mtxc->values.nlens = maxnnz;
    mtxc->rowinds.values = realloc(mtxc->rowinds.values, (maxnnz) * sizeof mtxc->rowinds);
    mtxc->colinds.values = realloc(mtxc->colinds.values, (maxnnz) * sizeof mtxc->colinds);
    mtxc->values.values = realloc(mtxc->values.values, (maxnnz) * sizeof mtxc->values);

    tnsIndex i;
    tnsIndex j;
    #pragma omp parallel for num_threads(NTHREADS) private(i,j)
    for(i = 0; i < mtxa->nnz; ++i){ 
        for(j = 0; j < mtxb->nnz; ++j){
            if(mtxa->rowinds.values[i] == mtxb->rowinds.values[j] && mtxa->colinds.values[i] == mtxb->colinds.values[j]){
                #pragma omp critical
                {
                    mtxc->rowinds.values[mtxc->nnz] = mtxa->rowinds.values[i];
                    mtxc->colinds.values[mtxc->nnz] = mtxa->colinds.values[i];
                    mtxc->values.values[mtxc->nnz] = mtxa->values.values[i] * mtxb->values.values[j];
                    ++mtxc->nnz;
                }
            } ///< if
        } ///< for j
    } ///< for i

    mtxc->rowinds.nlens = mtxc->nnz;
    mtxc->colinds.nlens = mtxc->nnz;
    mtxc->values.nlens = mtxc->nnz;
    return 0;
}


/*
*  稠密矩阵部分
*/
/// 稠密矩阵Hadamard积
int tnsHadamardDenmat(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb) {
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    for(tnsIndex i = 0; i < mtxa->nrows; ++i)
        for(tnsIndex j = 0; j < mtxa->ncols; ++j)
            mtxc->values.values[i * mtxa->stride + j] = mtxa->values.values[i * mtxa->stride + j] * mtxb->values.values[i * mtxb->stride + j];
    return 0;
}


/// 并行稠密矩阵Hadamard积
int tnsOmpHadamardDenmat(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb) {
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    #pragma omp parallel for num_threads(NTHREADS) 
    for(tnsIndex i = 0; i < mtxa->nrows; ++i)
        for(tnsIndex j = 0; j < mtxa->ncols; ++j)
            mtxc->values.values[i * mtxa->ncols + j] = mtxa->values.values[i * mtxa->ncols + j] * mtxb->values.values[i * mtxa->ncols + j];
    return 0;
}


/// 稠密矩阵Hadamard积（覆盖原空间存结果）
int tnsHadamardDenmat_rw(tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb){
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    for(tnsIndex i = 0; i < mtxa->nrows; ++i)
        for(tnsIndex j = 0; j < mtxa->ncols; ++j)
            mtxa->values.values[i * mtxa->stride + j] = mtxa->values.values[i * mtxa->stride + j] * mtxb->values.values[i * mtxb->stride + j];                                                                                                                                                                 
    return 0;
}


/// 并行稠密矩阵Hadamard积（覆盖原空间存结果）
int tnsOmpHadamardDenmat_rw(tnsDenseMatrix *mtxa, const tnsDenseMatrix *mtxb){
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    #pragma omp parallel for num_threads(NTHREADS)
    for(tnsIndex i = 0; i < mtxa->nrows; ++i)
        for(tnsIndex j = 0; j < mtxa->ncols; ++j)
            mtxa->values.values[i * mtxa->stride + j] = mtxa->values.values[i * mtxa->stride + j] * mtxb->values.values[i * mtxb->stride + j];                                                                                                                                                                 
    return 0;
}




/// 稀疏矩阵和稠密矩阵Hadamard积
int tnsHadamardDenSpamat(tnsSparseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsSparseMatrix *mtxb){
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    mtxc->rowinds.nlens = mtxb->nnz;
    mtxc->colinds.nlens = mtxb->nnz;
    mtxc->values.nlens = mtxb->nnz;
    mtxc->rowinds.values = realloc(mtxc->rowinds.values, (mtxb->nnz) * sizeof mtxc->rowinds);
    mtxc->colinds.values = realloc(mtxc->colinds.values, (mtxb->nnz) * sizeof mtxc->colinds);
    mtxc->values.values = realloc(mtxc->values.values, (mtxb->nnz) * sizeof mtxc->values);

    for(tnsIndex row_i = 0; row_i < mtxa->nrows; ++row_i){
        for(tnsIndex col_i = 0; col_i < mtxa->ncols; ++col_i){
            for(tnsIndex nnz_i = 0; nnz_i < mtxb->nnz; ++nnz_i){
                if(row_i == mtxb->rowinds.values[nnz_i] && col_i == mtxb->colinds.values[nnz_i]){
                    mtxc->rowinds.values[mtxc->nnz] = row_i;
                    mtxc->colinds.values[mtxc->nnz] = col_i;
                    mtxc->values.values[mtxc->nnz] = mtxa->values.values[row_i * mtxa->stride + col_i] * mtxb->values.values[nnz_i];
                    ++mtxc->nnz;
                }///< if
            }///< for nnz_i
        }///< for col_i
    }///< for row_i

    mtxc->rowinds.nlens = mtxc->nnz;
    mtxc->colinds.nlens = mtxc->nnz;
    mtxc->values.nlens = mtxc->nnz;
    return 0;
}

/// 并行稀疏矩阵和稠密矩阵Hadamard积
int tnsOmpHadamardDenSpamat(tnsSparseMatrix *mtxc, const tnsDenseMatrix *mtxa, const tnsSparseMatrix *mtxb){
    if(mtxa->nrows != mtxb->nrows || mtxa->ncols != mtxb->ncols || mtxa->nrows != mtxc->nrows || mtxa->ncols != mtxc->ncols){
        printf("维数不匹配\n");
        return 1;
    }
    
    mtxc->rowinds.values = realloc(mtxc->rowinds.values, (mtxb->nnz) * sizeof mtxc->rowinds);
    mtxc->colinds.values = realloc(mtxc->colinds.values, (mtxb->nnz) * sizeof mtxc->colinds);
    mtxc->values.values = realloc(mtxc->values.values, (mtxb->nnz) * sizeof mtxc->values);

    #pragma omp parallel for num_threads(4)
    for(tnsIndex row_i = 0; row_i < mtxa->nrows; ++row_i){
        for(tnsIndex col_i = 0; col_i < mtxa->ncols; ++col_i){
            for(tnsIndex nnz_i = 0; nnz_i < mtxb->nnz; ++nnz_i){
                if(row_i == mtxb->rowinds.values[nnz_i] && col_i == mtxb->colinds.values[nnz_i]){
                    mtxc->rowinds.values[mtxc->nnz] = row_i;
                    mtxc->colinds.values[mtxc->nnz] = col_i;
                    mtxc->values.values[mtxc->nnz] = mtxa->values.values[row_i * mtxa->stride + col_i] * mtxb->values.values[nnz_i];
                    ++mtxc->nnz;
                }///< if
            }///< for nnz_i
        }///< for col_i
    }///< for row_i
    return 0;
}