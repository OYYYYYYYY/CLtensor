#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

extern "C"
{
#include "TArm.h"
#include "timer.h"
#include <stdio.h>
}

// TTV
__global__ void tnsTTVKernel( 
    const tnsValue *X_val, const tnsValue *V_val, tnsValue *Y_val, 
    const tnsIndex X_nnz, const tnsIndex Y_nnz, const tnsIndex V_len)
{
    tnsIndex nnz_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(nnz_i < X_nnz){
        atomicAdd(Y_val + nnz_i / V_len, X_val[nnz_i] * V_val[nnz_i % V_len]);
    }
} 



int tnsHipTTVDentsr(tnsDenseTensor *dentsrY, const tnsDenseTensor *dentsrX, const tnsValueVector *vec, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= dentsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(dentsrX->ndims[copt_mode] != vec->nlens || dentsrY->ndims[copt_mode] != 1){
        printf("维度不匹配\n");
        return 1;
    }

    tnsValue* X_val;
    hipMalloc(&X_val, dentsrX->nnz * sizeof(tnsValue));
    hipMemcpy(X_val, dentsrX->values.values, dentsrX->nnz * sizeof(tnsValue), hipMemcpyHostToDevice);

    tnsValue* V_val;
    hipMalloc(&V_val, vec->nlens * sizeof(tnsValue));
    hipMemcpy(V_val, vec->values, vec->nlens * sizeof(tnsValue), hipMemcpyHostToDevice);

    tnsValue* Y_val;
    hipMalloc(&Y_val, dentsrY->nnz * sizeof(tnsValue));
    hipMemcpy(Y_val, dentsrY->values.values, dentsrY->nnz * sizeof(tnsValue), hipMemcpyHostToDevice);

    //dim3 dimBlock(16, 16);
    //dim3 dimGrid(32 / dimBlock.x, 32 / dimBlock.y);
    dim3 block(256);
    dim3 grid(1);
    grid.x = (dentsrX->nnz + block.x - 1) / block.x;

    Timer timer;
    timer_start(&timer);
    //hipLaunchKernelGGL(tnsTTVKernel, dimGrid, dimBlock, 0, 0,  
        // X_val, V_val, Y_val, 
        // dentsrX->nnz, dentsrY->nnz, vec->nlens);  
    hipLaunchKernelGGL(tnsTTVKernel, grid, block, 0, 0,  
        X_val, V_val, Y_val, 
        dentsrX->nnz, dentsrY->nnz, vec->nlens);  

    timer_stop(&timer);
    timer_print_sec(&timer, "DCU Dense TTV");

    hipMemcpy(dentsrY->values.values, Y_val, dentsrY->nnz * sizeof(tnsValue), hipMemcpyDeviceToHost);
    
    hipFree(X_val);
    hipFree(V_val);
    hipFree(Y_val);
    
    return 0;
}

int main(int agrc, char *argv[]){

    // 稠密TTV
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
    printf("nnz = %u\n", tensory.nnz);
    tnsFreeValueVector(&tensory.values);
    tnsNewValueVector(&tensory.values, tensory.nnz);
    printf("create tensory\n");
    
    tnsHipTTVDentsr(&tensory, &tensorx, &vvc, copt_mode);
    printf("finish single thread compution\n");
    
    FILE *fp_wttv = fopen("/public/software/apps/ghfund/ghfund202107013482/example1/result_dcu.log", "w");
    tnsDumpDenseTensor(&tensory, fp_wttv);
    printf("finish output\n");
    
    fclose(fp_wttv);
    tnsFreeDenseTensor(&tensorx);
    tnsFreeDenseTensor(&tensory);
    tnsFreeValueVector(&vvc);

    return 0;
}