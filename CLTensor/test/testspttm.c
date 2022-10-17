#include<stdio.h>
#include<stdlib.h>
#include "TArm.h"
#include "timer.h"

int main(int argc, char *argv[]){
    // 稀疏TTM
    
    Timer timer;
    
    tnsIndex copt_mode = 1;

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
    printf("create Y\n");

    timer_start(&timer);
    tnsTTMSpatsr(&tensory, &tensorx, &denmat, copt_mode);
    timer_stop(&timer);
    timer_print_sec(&timer, "single thread ttm");
    printf("finish compution\n");


    FILE *fp_w = fopen("/public/software/apps/ghfund/ghfund202107013482/example2/result_cpu.log", "w");
    tnsDumpSparseTensor(&tensory, fp_w);
    printf("finish output\n");
    fclose(fp_w);
    
    tnsFreeSparseTensor(&tensorx);
    tnsFreeSparseTensor(&tensory);
    tnsFreeDenseMatrix(&denmat);

    return 0;
}