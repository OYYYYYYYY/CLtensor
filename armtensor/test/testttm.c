
#include <TArm.h>
#include <timer.h>

#include <stdlib.h>
#include <stdio.h>
// #include <string.h>
#include "omp.h"

int main(void) {

    tnsSparseTensor spatsr_a;
    FILE* fp_r = fopen("tensor/3d_3_8.tns", "r");
    tnsLoadSparseTensor(&spatsr_a, fp_r);
    fclose(fp_r);

    tnsDenseMatrix denmat_b;
    FILE * fp_dense_mtx = fopen("tensor/dense_3_3.tns", "r");
    tnsLoadDenseMatrix(&denmat_b, fp_dense_mtx);
    fclose(fp_dense_mtx);

    tnsIndex copt_mode = 1;

    tnsDenseTensor dentsr_c;
    tnsNewDenseTensor(&dentsr_c, spatsr_a.nmodes, spatsr_a.ndims);
    tnsConstantDenseTensor(&dentsr_c, 0);

    tnsTTMSpatsr(&dentsr_c, &spatsr_a, &denmat_b, copt_mode);

    FILE * fp_w = fopen("tensor/result.tns", "w");
    tnsDumpDenseTensor(&dentsr_c, fp_w);
    fclose(fp_w);


    tnsFreeSparseTensor(&spatsr_a);
    tnsFreeDenseMatrix(&denmat_b);
    tnsFreeDenseTensor(&dentsr_c);


    // Timer test_timer;
    // timer_start(&test_timer);

    // tnsDenseMatrix mtx_dense;
    // FILE * fp_dense_mtx = fopen("tensor/dense.tns", "r");
    // tnsLoadDenseMatrix(&mtx_dense, fp_dense_mtx);

    // tnsDenseMatrix mtx_B;
    // tnsNewDenseMatrix(&mtx_B, mtx_dense.nrows, mtx_dense.ncols);
    // tnsCopyDenseMatrix(&mtx_B, &mtx_dense);

    // //test dense matrix Hadamard Product
    // tnsDenseMatrix mtxc_dense;
    // tnsNewDenseMatrix(&mtxc_dense, mtx_dense.nrows * mtx_B.nrows, mtx_dense.ncols * mtx_B.ncols);
    // printf("稠密矩阵Kronecker积 \n");
    // tnsKroneckerDenmtx(&mtxc_dense, &mtx_dense, &mtx_B);

    // FILE * fp_w = fopen("tensor/result.tns", "w");
    // tnsDumpDenseMatrix(&mtxc_dense, fp_w);


    // printf("释放稠密矩阵\n");
    // tnsFreeDenseMatrix(&mtx_dense);
    // tnsFreeDenseMatrix(&mtx_B);
    // tnsFreeDenseMatrix(&mtxc_dense);


    // fclose(fp_w);
    // fclose(fp_dense_mtx);
    // timer_stop(&test_timer);
    // timer_print_sec(&test_timer, "test_func");
    // timer_print_usec(&test_timer, "test_func");
    // printf("test_timer: %f\n",test_timer.seconds);

}