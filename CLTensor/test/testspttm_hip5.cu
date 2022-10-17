#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

extern "C"
{
#include "TArm.h"
#include "timer.h"

}

// SpTTM
__global__ void tnsTTMS1Kernel(
    const tnsValue *X_val, const tnsIndex *X_ndims, const tnsIndex *X_inds,
    const tnsIndex X_nnz, const tnsIndex X_nmodes, const tnsIndex step,
    const tnsValue *M_val, const tnsIndex M_nrows, const tnsIndex M_ncols, 
    const tnsIndex stride, tnsValue *V_val, const tnsIndex V_nlens, 
    const tnsIndex copt_mode)
{
    tnsIndex hash_index;
    tnsIndex hash_step;
    tnsIndex indexs = 1;
    tnsValue data_a;
    tnsValue data_b;
    tnsValue temp;
    // tnsIndex count = 0; 
    // tnsIndex k = 0;
    // // 计算中间结果values
    // for(tnsIndex nnz_i = 0; nnz_i < X_nnz; ++nnz_i){
    //     data_a = X_val[nnz_i];
    //     hash_index = 0;
    //     hash_step = 1;
    //     for(int mode_i = X_nmodes - 1; mode_i >= 0; --mode_i){
    //         if(mode_i != copt_mode){
    //             hash_index += X_inds[mode_i + nnz_i * X_nmodes] * hash_step;
    //             hash_step *= X_ndims[mode_i];
    //         }///< if
    //     }///< for mode_
    //     for(tnsIndex col_i = 0; col_i < M_ncols; ++col_i){
    //         data_b = M_val[X_inds[copt_mode + nnz_i * X_nmodes] * M_ncols + col_i];
    //         // test_val[nnz_i * X_nmodes + col_i] = data_b;
    //         indexs = col_i * step + hash_index;
    //         temp = X_val[nnz_i] * M_val[X_inds[copt_mode + nnz_i * X_nmodes] * M_ncols + col_i];
    //         V_val[indexs] += temp;
    //         test_val[k++] = data_a;
    //         test_val[k++] = data_b;
    //         test_val[k++] = temp;
    //         test_val[k++] = V_val[indexs];
    //     }///< for col_i
    // }///< for nnz_i
    // tnsIndex k = 0;
	// tnsIndex temp;
    tnsIndex nnz_i = blockDim.x * blockIdx.x + threadIdx.x;
    if(nnz_i < X_nnz){
        // data_a = X_val[nnz_i];
        hash_index = 0;
        hash_step = 1;
        for(int mode_i = X_nmodes - 1; mode_i >= 0; --mode_i){
            if(mode_i != copt_mode){
                hash_index += X_inds[mode_i + nnz_i * X_nmodes] * hash_step;
                hash_step *= X_ndims[mode_i];
            }///< if
        }///< for mode_
        for(tnsIndex col_i = 0; col_i < M_ncols; ++col_i){
            // data_b = M_val[X_inds[copt_mode + nnz_i * X_nmodes] * M_ncols + col_i];
            indexs = col_i * step + hash_index;
            V_val[indexs] = X_val[nnz_i] * M_val[X_inds[copt_mode + nnz_i * X_nmodes] * M_ncols + col_i];
        }///< for col_i
    }///< if nnz_i

    
}

__global__ void tnsTTMS2Kernel(
    tnsValue *Y_val, const tnsIndex *Y_ndims, tnsIndex *Y_inds,
    tnsIndex Y_nmodes, tnsValue *V_val, const tnsIndex V_nlens, 
    const tnsIndex copt_mode, const tnsIndex stride)
{
    
    tnsIndex hash_index;
    tnsIndex hash_step;
    tnsIndex indexs = 1;

    tnsIndex nnz_j = 0;
    for(tnsIndex nnz_i = 0; nnz_i < V_nlens; ++nnz_i){
        if(V_val[nnz_i] != 0){
            Y_val[nnz_j] = V_val[nnz_i];
            indexs = nnz_i;
            for(int mode_i = Y_nmodes - 1; mode_i >= 0; --mode_i){
                if(mode_i == copt_mode){
                    Y_inds[mode_i + nnz_j * Y_nmodes] = nnz_i / stride;
                }///< if(mode_i == copt_mode)
                else{
                    Y_inds[mode_i + nnz_j * Y_nmodes] = indexs % Y_ndims[mode_i];
                    indexs = indexs / Y_ndims[mode_i];
                }///< else
            }///< for mode_i
            ++nnz_j;
        }///< if(imdvec.values[nnz_i] != 0)
    }///< for nnz_i

} 



int tnsHipTTMSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsDenseMatrix *denmat, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(spatsrX->ndims[copt_mode] != denmat->nrows || spatsrY->ndims[copt_mode] != denmat->ncols){
        printf("维度不匹配\n");
        return 1;
    }

    // 获取spatsrY的松散上界
    tnsIndex length = 1;
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i){
        length  = length * spatsrY->ndims[i];
    } 
    // 获取步长strides
    tnsIndex strides = length / spatsrY->ndims[copt_mode];
    
    // 创建向量imdvec存储中间结果
    tnsValueVector imdvec;
    tnsNewValueVector(&imdvec, length);
    tnsConstantValueVector(&imdvec, 0);

    // 创建向量indvecX存储张量X的索引（按每一个非零元的顺序）
    tnsIndexVector indvecX;
    tnsNewIndexVector(&indvecX, spatsrX->nnz * spatsrX->nmodes);
    for(tnsIndex i = 0; i < spatsrX->nmodes; ++i)
        for(tnsIndex j = 0; j < spatsrX->nnz; ++j)
            indvecX.values[j * spatsrX->nmodes + i] = spatsrX->inds[i].values[j];
        
    printf("Stage 1 : ready to move data to DCU\n");

    // 将host端数据复制到device端
    tnsValue* X_val;
    hipMalloc(&X_val, spatsrX->nnz * sizeof(tnsValue));
    hipMemcpy(X_val, spatsrX->values.values, spatsrX->nnz * sizeof(tnsValue), hipMemcpyHostToDevice);

    tnsIndex* X_ndims;
    hipMalloc(&X_ndims, spatsrX->nmodes * sizeof(tnsIndex));
    hipMemcpy(X_ndims, spatsrX->ndims, spatsrX->nmodes * sizeof(tnsIndex), hipMemcpyHostToDevice);
    
    tnsIndex* X_inds;
    hipMalloc(&X_inds, indvecX.nlens * sizeof(tnsIndex));
    hipMemcpy(X_inds, indvecX.values, indvecX.nlens * sizeof(tnsIndex), hipMemcpyHostToDevice);

    tnsValue* M_val;
    hipMalloc(&M_val, denmat->nrows * denmat->ncols * sizeof(tnsValue));
    hipMemcpy(M_val, denmat->values.values, denmat->nrows * denmat->ncols * sizeof(tnsValue), hipMemcpyHostToDevice);

    tnsValue* V_val;
    hipMalloc(&V_val, imdvec.nlens * sizeof(tnsValue));
    hipMemcpy(V_val, imdvec.values, imdvec.nlens * sizeof(tnsValue), hipMemcpyHostToDevice);

    // 释放向量indvecX
    tnsFreeIndexVector(&indvecX);

    // 启动kernel函数tnsTTMS1Kernel计算中间结果
	printf("Stage 1 : finish move\n");
    
    // dim3 dimBlock(32, 32);
    // dim3 dimGrid(256 / dimBlock.x, 256 / dimBlock.y);
    dim3 block(256);
    dim3 grid(1);
    grid.x = (spatsrX->nnz + block.x - 1) / block.x;

    Timer timer;
    timer_start(&timer);
    // hipLaunchKernelGGL(tnsTTMS1Kernel, dimGrid, dimBlock, 0, 0, 
    //     X_val, X_ndims, X_inds,
    //     spatsrX->nnz, spatsrX->nmodes, strides, 
    //     M_val, denmat->nrows, denmat->ncols, 
    //     denmat->stride, V_val, imdvec.nlens, 
    //     copt_mode);
    hipLaunchKernelGGL(tnsTTMS1Kernel, grid, block, 0, 0, 
        X_val, X_ndims, X_inds,
        spatsrX->nnz, spatsrX->nmodes, strides, 
        M_val, denmat->nrows, denmat->ncols, 
        denmat->stride, V_val, imdvec.nlens, 
        copt_mode);

    timer_stop(&timer);
    // timer_print_sec(&timer, "DCU ttm Stage 1");
    tnsValue time = timer.seconds;
    
    // 将中间结果从device端复制到host端
    hipMemcpy(imdvec.values, V_val, imdvec.nlens * sizeof(tnsValue), hipMemcpyDeviceToHost);

    // 统计中间结果（结果张量）的非零元数
    tnsIndex count = 0;
    for(tnsIndex i = 0; i < imdvec.nlens; ++i)
        if(imdvec.values[i] != 0)
            ++count;
    spatsrY->nnz = count;
    // printf("nnz of spatsrY after computation = %u\n", spatsrY->nnz);
    
    // 释放device端内存
    hipFree(X_val);
    hipFree(X_ndims);
    hipFree(X_inds);
    hipFree(M_val);
    
    //通过准确的非零元数重新分配spatsrY的values和inds在host端的空间
    tnsFreeValueVector(&spatsrY->values);
    tnsNewValueVector(&spatsrY->values, spatsrY->nnz);
    for(tnsIndex mode = 0; mode < spatsrY->nmodes; ++mode){
        tnsFreeIndexVector(&spatsrY->inds[mode]);
        tnsNewIndexVector(&spatsrY->inds[mode], spatsrY->nnz);
    }

    // 创建向量indvecX存储张量X的索引（按每一个非零元的顺序）
    tnsIndexVector indvec;
    tnsNewIndexVector(&indvec, spatsrY->nnz * spatsrY->nmodes);
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i)
        for(tnsIndex j = 0; j < spatsrY->nnz; ++j)
            indvec.values[j * spatsrY->nmodes + i] = spatsrY->inds[i].values[j];

    printf("Stage 2 : ready to move data to DCU\n");

    // 将host端数据复制到device端
    tnsValue* Y_val;
    hipMalloc(&Y_val, spatsrY->nnz * sizeof(tnsValue));
    hipMemcpy(Y_val, spatsrY->values.values, spatsrY->nnz * sizeof(tnsValue), hipMemcpyHostToDevice);

    tnsIndex* Y_ndims;
    hipMalloc(&Y_ndims, spatsrY->nmodes * sizeof(tnsIndex));
    hipMemcpy(Y_ndims, spatsrY->ndims, spatsrY->nmodes * sizeof(tnsIndex), hipMemcpyHostToDevice);

    tnsIndex* Y_inds;
    hipMalloc(&Y_inds, spatsrY->nnz * spatsrY->nmodes * sizeof(tnsIndex));
    hipMemcpy(Y_inds, indvec.values, indvec.nlens * sizeof(tnsIndex), hipMemcpyHostToDevice);

    hipMemcpy(V_val, imdvec.values, imdvec.nlens * sizeof(tnsValue), hipMemcpyHostToDevice);
    
    // 启动kernel函数tnsTTMS2Kernel将中间结果写回处理到结果张量
    printf("Stage 2 : finish move\n");
    timer_restart(&timer);
    hipLaunchKernelGGL(tnsTTMS2Kernel, 64, 512, 0, 0, 
        Y_val, Y_ndims, Y_inds,
        spatsrY->nmodes, V_val, imdvec.nlens, 
        copt_mode, strides);

    timer_stop(&timer);
    // timer_print_sec(&timer, "DCU ttm Stage 2");
    time += timer.seconds;
    printf("Total time is %.6f\n", time);
    // 创建向量indvecY存储张量spatsrY的索引
    tnsIndexVector indvecY;
    tnsNewIndexVector(&indvecY, spatsrY->nnz * spatsrY->nmodes);

    // 将数据从device端复制到host端
    hipMemcpy(spatsrY->values.values, Y_val, spatsrY->nnz * sizeof(tnsValue), hipMemcpyDeviceToHost);
    hipMemcpy(indvecY.values, Y_inds, spatsrY->nnz * spatsrY->nmodes * sizeof(tnsIndex), hipMemcpyDeviceToHost);

    // 将索引向量indvecY写回到spatsrY的inds中
    for(tnsIndex i = 0; i < spatsrY->nnz; ++i)
        for(tnsIndex j = 0; j < spatsrY->nmodes; ++j)
            spatsrY->inds[j].values[i] = indvecY.values[i * spatsrY->nmodes + j];
    tnsFreeIndexVector(&indvecY);

    hipFree(V_val);
    hipFree(Y_val);
    hipFree(Y_ndims);
    hipFree(Y_inds);
    
    return 0;
}

int main(int agrc, char *argv[]){

    tnsIndex copt_mode = 1;

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
    printf("create tensory\n");

    tnsHipTTMSpatsr(&tensory, &tensorx, &denmat, copt_mode);
    printf("finish compution\n");
    
    FILE *fp_w = fopen("/public/software/apps/ghfund/ghfund202107013482/example2/result_dcu5.log", "w");
    tnsDumpSparseTensor(&tensory, fp_w);
    printf("finish output\n");
    fclose(fp_w);
    
    tnsFreeSparseTensor(&tensorx);
    tnsFreeSparseTensor(&tensory);
    tnsFreeDenseMatrix(&denmat);

    return 0;
}