#include <TArm.h>
#include <stdlib.h>
#include <string.h>
#define nthreads 4

int tnsTTMSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsDenseMatrix *denmat, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= spatsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(spatsrX->ndims[copt_mode] != denmat->nrows || spatsrY->ndims[copt_mode] != denmat->ncols){
        printf("维度不匹配\n");
        return 1;
    }

    tnsIndex hash_index = 0;
    tnsIndex hash_step = 1;
    tnsIndex indexs = 1;
    tnsValue data_a;
    tnsValue data_b;
    tnsIndex count = 0;
    tnsIndex length = 1;


    // 获取spatsrY的松散上界
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i){
        length  = length * spatsrY->ndims[i];
    } 
    // 获取步长strides
    tnsIndex strides = length / spatsrY->ndims[copt_mode];
    
    // 创建Valuevacter datavec存储中间值
    tnsValueVector datavec;
    tnsNewValueVector(&datavec, length);

    // 计算中间结果values
    for(tnsIndex nnz_i = 0; nnz_i < spatsrX->nnz; ++nnz_i){
        data_a = spatsrX->values.values[nnz_i];
        hash_index = 0;
        hash_step = 1;

        for(int mode_i = spatsrX->nmodes - 1; mode_i >= 0; --mode_i){
            if(mode_i != copt_mode){
                 hash_index += spatsrX->inds[mode_i].values[nnz_i] * hash_step;
                 hash_step *= spatsrX->ndims[mode_i];
            }///< if
        }///< for mode_i

        for(tnsIndex col_i = 0; col_i < denmat->ncols; ++col_i){
            data_b = denmat->values.values[spatsrX->inds[copt_mode].values[nnz_i] * denmat->stride + col_i];

            indexs = col_i * strides + hash_index;
            datavec.values[indexs] += data_a * data_b;
        }///< for col_i
    }///< for nnz_i


    for(tnsIndex i = 0; i < datavec.nlens; ++i)
        if(datavec.values[i] != 0)
            ++count;
    spatsrY->nnz = count;

    //通过准确的非零元数重新分配spatsrY的values和inds空间
    tnsFreeValueVector(&spatsrY->values);
    tnsNewValueVector(&spatsrY->values, spatsrY->nnz);
    for(tnsIndex mode = 0; mode < spatsrY->nmodes; ++mode){
        tnsFreeValueVector(&spatsrY->inds[mode]);
        tnsNewIndexVector(&spatsrY->inds[mode], spatsrY->nnz);
    }

    tnsIndex nnz_j = 0;
    for(tnsIndex nnz_i = 0; nnz_i < datavec.nlens; ++nnz_i){
        if(datavec.values[nnz_i] != 0){
            spatsrY->values.values[nnz_j] = datavec.values[nnz_i];
            indexs = nnz_i;
            for(int mode_i = spatsrY->nmodes - 1; mode_i >= 0; --mode_i){
                if(mode_i == copt_mode){
                    spatsrY->inds[mode_i].values[nnz_j] = nnz_i / strides;
                }///< if(mode_i == copt_mode)
                else{
                    spatsrY->inds[mode_i].values[nnz_j]  = indexs % spatsrY->ndims[mode_i];
                    indexs = indexs / spatsrY->ndims[mode_i];
                }///< else
            }///< for mode_i
            ++nnz_j;
        }///< if(datavec.values[nnz_i] != 0)
    }///< for nnz_i

    tnsFreeValueVector(&datavec);
    return 0;
}

int tnsOMPTTMSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsDenseMatrix *denmat, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= spatsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(spatsrX->ndims[copt_mode] != denmat->nrows || spatsrY->ndims[copt_mode] != denmat->ncols){
        printf("维度不匹配\n");
        return 1;
    }

    tnsIndex hash_index = 0;
    tnsIndex hash_step = 1;
    tnsIndex indexs = 1;
    tnsValue data_a;
    tnsValue data_b;
    tnsIndex count = 0;
    tnsIndex length = 1;


    // 获取spatsrY的松散上界
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i){
        length  = length * spatsrY->ndims[i];
    } 
    // 获取步长strides
    tnsIndex strides = length / spatsrY->ndims[copt_mode];
    
    // 创建Valuevacter datavec存储中间值
    tnsValueVector datavec;
    tnsNewValueVector(&datavec, length);

    // 计算中间结果values
    #pragma omp parallel for num_threads(nthreads)
    for(tnsIndex nnz_i = 0; nnz_i < spatsrX->nnz; ++nnz_i){
        data_a = spatsrX->values.values[nnz_i];
        hash_index = 0;
        hash_step = 1;

        for(int mode_i = spatsrX->nmodes - 1; mode_i >= 0; --mode_i){
            if(mode_i != copt_mode){
                 hash_index += spatsrX->inds[mode_i].values[nnz_i] * hash_step;
                 hash_step *= spatsrX->ndims[mode_i];
            }///< if
        }///< for mode_i

        for(tnsIndex col_i = 0; col_i < denmat->ncols; ++col_i){
            data_b = denmat->values.values[spatsrX->inds[copt_mode].values[nnz_i] * denmat->stride + col_i];

            indexs = col_i * strides + hash_index;
            datavec.values[indexs] += data_a * data_b;
        }///< for col_i
    }///< for nnz_i


    for(tnsIndex i = 0; i < datavec.nlens; ++i)
        if(datavec.values[i] != 0)
            ++count;
    spatsrY->nnz = count;

    //通过准确的非零元数重新分配spatsrY的values和inds空间
    tnsFreeValueVector(&spatsrY->values);
    tnsNewValueVector(&spatsrY->values, spatsrY->nnz);
    for(tnsIndex mode = 0; mode < spatsrY->nmodes; ++mode){
        tnsFreeValueVector(&spatsrY->inds[mode]);
        tnsNewIndexVector(&spatsrY->inds[mode], spatsrY->nnz);
    }

    tnsIndex nnz_j = 0;
    #pragma omp parallel for num_threads(nthreads)
    for(tnsIndex nnz_i = 0; nnz_i < datavec.nlens; ++nnz_i){
        if(datavec.values[nnz_i] != 0){
            spatsrY->values.values[nnz_j] = datavec.values[nnz_i];
            indexs = nnz_i;
            for(int mode_i = spatsrY->nmodes - 1; mode_i >= 0; --mode_i){
                if(mode_i == copt_mode){
                    spatsrY->inds[mode_i].values[nnz_j] = nnz_i / strides;
                }///< if(mode_i == copt_mode)
                else{
                    spatsrY->inds[mode_i].values[nnz_j]  = indexs % spatsrY->ndims[mode_i];
                    indexs = indexs / spatsrY->ndims[mode_i];
                }///< else
            }///< for mode_i
            ++nnz_j;
        }///< if(datavec.values[nnz_i] != 0)
    }///< for nnz_i

    tnsFreeValueVector(&datavec);
    return 0;
}