#include<stdio.h>
#include<stdlib.h>
#include "omp.h"
#include "TArm.h"

int main(int argc, char *argv[]) {
    TImer timer;
    timer_reset(&timer);

    // 稀疏TTV
    tnsIndex copt_mode = 0;

    // 加载张量tensorx
    tnsSparseTensor tensorx;
    FILE *fp_x = fopen(argv[1], "r");
    tnsLoadSparseTensor(&tensorx, fp_x);
    fclose(fp_x);
    printf("load tensorx\n");
    
    // 加载向量vvc
    tnsValueVector vvc;
    FILE *fp_v = fopen(argv[2], "r");
    tnsNewValueVector(&vvc, tensorx.ndims[copt_mode]);
    tnsLoadValueVector(&vvc, fp_v);
    fclose(fp_v);
    printf("load vector\n");

    tnsSparseTensor tensory;
    tnsNewSparseTensor(&tensory, tensorx.ndims, tensorx.nmodes, tensorx.nnz);
    tensory.ndims[copt_mode] = 1;
    printf("create tensory\n");
    
    timer_start(&timer);
    tnsTTVSpatsr(&tensory, &tensorx, &vvc, copt_mode);
    timer_stop(&timer);
    timer_print_sec(&timer, "single thread: ");
    printf("finish single thread compution\n");
    
    
    FILE *fp_wttv = fopen("oydata/spattv.tns", "w");
    tnsDumpSparseTensor(&tensory, fp_wttv);
    printf("finish output\n");


    tnsSparseTensor tensorz;
    tnsNewSparseTensor(&tensorz, tensorx.ndims, tensorx.nmodes, tensorx.nnz);
    tensorz.ndims[copt_mode] = 1;
    printf("create tensorz\n");

    timer_reset(&timer);
    timer_start(&timer);
    tnsOmpTTVSpatsr(&tensorz, &tensorx, &vvc, copt_mode);
    timer_stop(&timer);
    timer_print_sec(&tiemr, "multiple threads: ");
    printf("finish compution\n");

    tnsDumpSparseTensor(&tensorz, fp_wttv);
    printf("finish output\n");
    
    fclose(fp_wttv);
    tnsFreeSparseTensor(&tensorx);
    tnsFreeSparseTensor(&tensory);
    tnsFreeSparseTensor(&tensorz);
    tnsFreeValueVector(&vvc);

    return 0;
}