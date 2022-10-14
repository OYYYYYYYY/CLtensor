#include "omp.h"
#include <TArm.h>
#include <stdlib.h>

#define nthreads 4

// 稀疏张量矩阵化，将cutPoint前面的模态作为行，后面模态作为列。
int tnsMatricizationSpatsr(tnsSparseMatrix *mtx, const tnsSparseTensor *tsr, tnsIndex cutPoint){
    mtx->nnz = tsr->nnz;
    mtx->rowinds.nlens = mtx->nnz;
    mtx->colinds.nlens = mtx->nnz;
    mtx->values.nlens = mtx->nnz;
    mtx->rowinds.values = realloc(mtx->rowinds.values, (mtx->nnz) * sizeof mtx->rowinds);
    mtx->colinds.values = realloc(mtx->colinds.values, (mtx->nnz) * sizeof mtx->colinds);
    mtx->values.values = realloc(mtx->values.values, (mtx->nnz) * sizeof mtx->values);
    
    tnsIndex hash_rowstep = 1;
    tnsIndex hash_colstep = 1;
    tnsIndex hash_rowinds = 0;
    tnsIndex hash_colinds = 0; 
    for(tnsIndex nnz_i = 0; nnz_i < tsr->nnz; ++nnz_i){
        hash_rowstep = 1;
        hash_colstep = 1;
        hash_rowinds = 0;
        hash_colinds = 0;
        for(int mode_i = cutPoint - 1; mode_i >= 0; --mode_i){
            hash_rowinds += tsr->inds[mode_i].values[nnz_i] * hash_rowstep;
            hash_rowstep *= tsr->ndims[mode_i]; 
        }///< for mode_i
        for(int mode_j = tsr->nmodes - 1; mode_j >= cutPoint; --mode_j){
            hash_colinds += tsr->inds[mode_j].values[nnz_i] * hash_colstep;
            hash_colstep *= tsr->ndims[mode_j];
        }///< for mode_j
        
        mtx->rowinds.values[nnz_i] = hash_rowinds;
        mtx->colinds.values[nnz_i] = hash_colinds;
        mtx->values.values[nnz_i] = tsr->values.values[nnz_i];
    }///< for nnz_i
    
    return 0;
}

// 并行稀疏张量矩阵化，将cutPoint前面的模态作为行，后面模态作为列。
int tnsOmpMatricizationSpatsr(tnsSparseMatrix *mtx, const tnsSparseTensor *tsr, tnsIndex cutPoint){
    mtx->nnz = tsr->nnz;
    mtx->rowinds.nlens = mtx->nnz;
    mtx->colinds.nlens = mtx->nnz;
    mtx->values.nlens = mtx->nnz;
    mtx->rowinds.values = realloc(mtx->rowinds.values, (mtx->nnz) * sizeof mtx->rowinds);
    mtx->colinds.values = realloc(mtx->colinds.values, (mtx->nnz) * sizeof mtx->colinds);
    mtx->values.values = realloc(mtx->values.values, (mtx->nnz) * sizeof mtx->values);
    
    tnsIndex hash_rowstep = 1;
    tnsIndex hash_colstep = 1;
    tnsIndex hash_rowinds = 0;
    tnsIndex hash_colinds = 0; 

    #pragma omp parallel for num_threads(nthreads)
    for(tnsIndex nnz_i = 0; nnz_i < tsr->nnz; ++nnz_i){
        hash_rowstep = 1;
        hash_colstep = 1;
        hash_rowinds = 0;
        hash_colinds = 0;
        for(int mode_i = cutPoint - 1; mode_i >= 0; --mode_i){
            hash_rowinds += tsr->inds[mode_i].values[nnz_i] * hash_rowstep;
            hash_rowstep *= tsr->ndims[mode_i]; 
        }///< for mode_i
        for(int mode_j = tsr->nmodes - 1; mode_j >= cutPoint; --mode_j){
            hash_colinds += tsr->inds[mode_j].values[nnz_i] * hash_colstep;
            hash_colstep *= tsr->ndims[mode_j];
        }///< for mode_j
        
        mtx->rowinds.values[nnz_i] = hash_rowinds;
        mtx->colinds.values[nnz_i] = hash_colinds;
        mtx->values.values[nnz_i] = tsr->values.values[nnz_i];
    }///< for nnz_i
    
    return 0;
}
