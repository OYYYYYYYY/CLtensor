#include<stdio.h>
#include<stdlib.h>
#include "omp.h"
#include "TArm.h"

int main(void) {

    // test hadamard
    
    tnsDenseMatrix mtxA;
    tnsDenseMatrix mtxB;
    tnsDenseMatrix mtxC;
    FILE *fp_dense_w = fopen("oydata/dehadamard.tns", "w");
    FILE *fp_dense_wrw = fopen("oydata/rwdehadamard.tns", "w");

    tnsNewDenseMatrix(&mtxA, 50, 50);
    tnsConstantDenseMatrix(&mtxA, 2);
    tnsNewDenseMatrix(&mtxB, 50, 50);
    tnsConstantDenseMatrix(&mtxB, 3);
    tnsNewDenseMatrix(&mtxC, 50, 50);
    /// 稠密哈达玛积(另开空间)
    tnsHadamardDenmat(&mtxC, &mtxA, &mtxB);
    tnsDumpDenseMatrix(&mtxC, fp_dense_w);
    tnsOmpHadamardDenmat(&mtxC, &mtxA, &mtxB);
    tnsDumpDenseMatrix(&mtxC, fp_dense_w);
    /// 稠密哈达玛积(覆盖原空间)
    tnsHadamardDenmat_rw(&mtxA, &mtxB);
    tnsDumpDenseMatrix(&mtxA, fp_dense_wrw);
    tnsOmpHadamardDenmat_rw(&mtxA, &mtxB);
    tnsDumpDenseMatrix(&mtxA, fp_dense_wrw);

    tnsFreeDenseMatrix(&mtxA);
    tnsFreeDenseMatrix(&mtxB);
    tnsFreeDenseMatrix(&mtxC);

    fclose(fp_dense_w);
    fclose(fp_dense_wrw);
    
    // 稀疏哈达玛

    tnsSparseMatrix mtxSA;
    tnsSparseMatrix mtxSB;
    tnsSparseMatrix mtxSC;
    tnsSparseMatrix mtxSD; 

    FILE *fp_sp_a = fopen("tensor/sparse.mtx", "r");
    FILE *fp_sp_b = fopen("oydata/sparse.mtx", "r");
    FILE *fp_sp_w = fopen("oydata/sphadamard.tns", "w");

    tnsLoadSparseMatrix(&mtxSA, fp_sp_a);
    tnsLoadSparseMatrix(&mtxSB, fp_sp_b);
    tnsNewSparseMatrix(&mtxSC, 3, 3);
    tnsNewSparseMatrix(&mtxSD, 3, 3);
    // 稀疏哈达玛
    tnsHadamardSpamat(&mtxSC, &mtxSA, &mtxSB);
    tnsDumpSparseMatrix(&mtxSC, fp_sp_w);
    
    tnsOmpHadamardSpamat(&mtxSD, &mtxSA, &mtxSB);
    tnsDumpSparseMatrix(&mtxSD, fp_sp_w);
    
    fclose(fp_sp_a);
    fclose(fp_sp_b);
    fclose(fp_sp_w);

    tnsFreeSparseMatrix(&mtxSA);
    tnsFreeSparseMatrix(&mtxSB);
    tnsFreeSparseMatrix(&mtxSC);
    tnsFreeSparseMatrix(&mtxSD);
    
    tnsDenseMatrix mtxAA;
    FILE *fp_ra = fopen("oydata/dense.mtx", "r");
    tnsLoadDenseMatrix(&mtxAA, fp_ra);
    fclose(fp_ra);

    tnsSparseMatrix mtxBB;
    FILE *fp_rb = fopen("oydata/sparse.mtx", "r");
    tnsLoadSparseMatrix(&mtxBB, fp_rb);
    fclose(fp_rb);

    tnsSparseMatrix mtxCC;
    FILE *fp_ww = fopen("oydata/denspahadamard.mtx", "w");
    tnsNewSparseMatrix(&mtxCC, mtxAA.nrows, mtxAA.ncols);
    tnsHadamardDenSpamat(&mtxCC, &mtxAA, &mtxBB);
    tnsDumpSparseMatrix(&mtxCC, fp_ww);
    
    tnsSparseMatrix mtxDD;
    tnsNewSparseMatrix(&mtxDD, mtxAA.nrows, mtxAA.ncols);
    tnsOmpHadamardDenSpamat(&mtxDD, &mtxAA, &mtxBB);
    tnsDumpSparseMatrix(&mtxDD, fp_ww);
    fclose(fp_ww);

    tnsFreeDenseMatrix(&mtxAA);
    tnsFreeSparseMatrix(&mtxBB);
    tnsFreeSparseMatrix(&mtxCC);
    tnsFreeSparseMatrix(&mtxDD);


    return 0;
}