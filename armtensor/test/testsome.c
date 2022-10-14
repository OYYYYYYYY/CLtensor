
#include <TArm.h>
#include <timer.h>

#include <stdlib.h>
#include <stdio.h>
// #include <string.h>
#include "omp.h"

int main(void) {

    Timer test_timer;
    timer_start(&test_timer);
    tnsDenseMatrix mtx_dense;
    FILE * fp_dense_mtx = fopen("tensor/dense.tns", "r");
    tnsLoadDenseMatrix(&mtx_dense, fp_dense_mtx);

    tnsDenseMatrix mtx_B;
    tnsNewDenseMatrix(&mtx_B, mtx_dense.nrows, mtx_dense.ncols);
    tnsCopyDenseMatrix(&mtx_B, &mtx_dense);

    //test dense matrix Hadamard Product
    tnsDenseMatrix mtxc_dense;
    tnsNewDenseMatrix(&mtxc_dense, mtx_dense.nrows * mtx_B.nrows, mtx_dense.ncols * mtx_B.ncols);
    printf("稠密矩阵Kronecker积 \n");
    tnsKroneckerDenmtx(&mtxc_dense, &mtx_dense, &mtx_B);

    FILE * fp_w = fopen("tensor/result.tns", "w");
    tnsDumpDenseMatrix(&mtxc_dense, fp_w);


    printf("释放稠密矩阵\n");
    tnsFreeDenseMatrix(&mtx_dense);
    tnsFreeDenseMatrix(&mtx_B);
    tnsFreeDenseMatrix(&mtxc_dense);


    fclose(fp_w);
    fclose(fp_dense_mtx);
    timer_stop(&test_timer);
    timer_print_sec(&test_timer, "test_func");
    timer_print_usec(&test_timer, "test_func");
    // printf("test_timer: %f\n",test_timer.seconds);



    // tnsIndex len = 10;
    // int a[10] = {0,0,0,0,0,0,0,0,0,0};
    // tnsValue value = 5;
    // printf("%f \n", value);
    // tns_CheckOSError(0 , "tns New1"); 
    // tns_CheckOSError(1 , "tns New2"); 
    // tns_CheckError(0 , "sptMatrixMulMatrix", "维度有问题！");
    // tns_CheckError(1 , "sptMatrixMulMatrix", "维度有问题！");
    
    // // #pragma omp parallel for
    // for(int i = 0; i < 10; i++){
    //     // a[i] = 1;
    //     printf("%d ", i);
    // } 

}