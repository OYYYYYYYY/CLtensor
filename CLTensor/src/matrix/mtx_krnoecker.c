#include <TArm.h>
#include <stdlib.h>
// #include <string.h>


/// 稠密矩阵Kronecker 积
int tnsKroneckerDenmtx(tnsDenseMatrix *mtxc, const tnsDenseMatrix *mtxa, tnsDenseMatrix *mtxb){
    // 输入参数检查
    tns_CheckError((mtxc->nrows == 0 || mtxc->ncols == 0 || mtxa->nrows == 0 || mtxa->ncols == 0 || mtxb->nrows == 0 || mtxb->ncols == 0), "tnsKroneckerDenmtx", "输入与输出矩阵维度为0，不可直接使用！");

    tns_CheckError((mtxc->nrows != (mtxa->nrows * mtxb->nrows) || mtxc->ncols != (mtxa->ncols * mtxb->ncols)), "tnsKroneckerDenmtx", "输入与输出维度大小不匹配！");



    // 计算过程
    tnsIndex startRow;
	tnsIndex startCol;
    tnsValue first_mtx;
    for(tnsIndex i = 0; i < mtxa->nrows; i++){
        for(tnsIndex j = 0; j < mtxa->ncols; j++){
            startRow = i * mtxb->nrows;
	        startCol = j * mtxb->ncols;
            first_mtx = mtxa->values.values[i * mtxa->stride + j];
            // printf("first_mtx %d,%d: %f \n", i,j,first_mtx);
            for(tnsIndex m = 0; m < mtxb->nrows; m++){
                for(tnsIndex n = 0; n < mtxb->ncols; n++){
                    mtxc->values.values[(startRow + m) * mtxc->stride + startCol + n] = first_mtx * mtxb->values.values[m * mtxb->stride + n];
                    // matrixC[startRow + m][startCol + n] = matrixA[i][j] * matrixB[m][n];
                } ///< for n
            } ///< for m

        } ///< for j
    } ///< for i

    return 0;

}

/// 稀疏矩阵Kronecker 积
int tnsKroneckerSpamtx(tnsSparseMatrix *mtxc, const tnsSparseMatrix *mtxa, tnsSparseMatrix *mtxb){
    // 输入参数检查
    tns_CheckError((mtxc->nrows == 0 || mtxc->ncols == 0 || mtxa->nrows == 0 || mtxa->ncols == 0 || mtxb->nrows == 0 || mtxb->ncols == 0), "tnsKroneckerDenmtx", "输入与输出矩阵维度为0，不可直接使用！");

    tns_CheckError((mtxc->nrows != (mtxa->nrows * mtxb->nrows) || mtxc->ncols != (mtxa->ncols * mtxb->ncols)), "tnsKroneckerDenmtx", "输入与输出维度大小不匹配！");
    

    // 计算过程
    for(tnsIndex i = 0; i < mtxa->nnz; i++){
        for(tnsIndex j = 0; j < mtxb->nnz; j++){
            j++;

        } ///< for j
    } ///< for i
    return 0;
}
