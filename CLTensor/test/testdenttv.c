#include<stdio.h>
#include<stdlib.h>
#include "TArm.h"
#include "timer.h"

int main(int argc, char *argv[]) {
    Timer timer;

    // 稀疏TTV
    tnsIndex copt_mode = 2;

    // 加载张量tensorx
    tnsDenseTensor tensorx;
    FILE *fp_x = fopen(argv[1], "r");
    tnsLoadDenseTensor(&tensorx, fp_x);
    fclose(fp_x);
    printf("load tensorx\n");
    
    // 加载向量vvc
    tnsValueVector vvc;
    FILE *fp_v = fopen(argv[2], "r");
    tnsNewValueVector(&vvc, tensorx.ndims[copt_mode]);
    tnsLoadValueVector(&vvc, fp_v);
    fclose(fp_v);
    printf("load vector\n");

    tnsDenseTensor tensory;
    tnsNewDenseTensor(&tensory, tensorx.nmodes, tensorx.ndims);
    tensory.ndims[copt_mode] = 1;
    tensory.nnz = tensory.nnz / tensorx.ndims[copt_mode];
    tnsFreeValueVector(&tensory.values);
    tnsNewValueVector(&tensory.values, tensory.nnz);
    // printf("nnz = %u\n", tensory.nnz);
    printf("create tensory\n");
    
    timer_start(&timer);
    tnsTTVDentsr(&tensory, &tensorx, &vvc, copt_mode);
    timer_stop(&timer);
    timer_print_sec(&timer, "single thread TTV ");
    printf("finish single thread compution\n");
     
    FILE *fp_wttv = fopen("/public/software/apps/ghfund/ghfund202107013482/example1/result_cpu.log", "w");
    tnsDumpDenseTensor(&tensory, fp_wttv);
    printf("finish output\n");
    
    fclose(fp_wttv);
    tnsFreeDenseTensor(&tensorx);
    tnsFreeDenseTensor(&tensory);
    tnsFreeValueVector(&vvc);

    return 0;
}