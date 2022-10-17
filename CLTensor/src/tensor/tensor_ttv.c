#include "omp.h"
#include <TArm.h>
#include <stdlib.h>
#include "timer.h"

#define nthreads 4
/*
* 稀疏张量
*/
//  Sparse tensor times a vector (SpTTV)
int tnsTTVSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsValueVector *vec, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= spatsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(spatsrX->ndims[copt_mode] != vec->nlens || spatsrY->ndims[copt_mode] != 1){
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

    // 获取spatsrY的松散上界并创建Valuevacter datavec存储中间值
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i){
        length  = length * spatsrY->ndims[i];
    } 

    tnsValueVector datavec;
    tnsNewValueVector(&datavec, length);

    // 计算values并统计spatsrY的非零元数
    for(tnsIndex nnz_i = 0; nnz_i < spatsrX->nnz; ++nnz_i){
        data_a = spatsrX->values.values[nnz_i];
        data_b = vec->values[spatsrX->inds[copt_mode].values[nnz_i]];
        hash_index = 0;
        hash_step = 1;
        // 计算spatsrY的values以及每个值对应的位置hash_index
        for(int mode_i = spatsrX->nmodes - 1; mode_i >= 0; --mode_i){
            if(mode_i != copt_mode){
                hash_index += spatsrX->inds[mode_i].values[nnz_i] * hash_step;
                hash_step *= spatsrX->ndims[mode_i];
            }///< if
        }///< for mode_i
        datavec.values[hash_index] += data_a * data_b;
    }///< for nnz_i

    for(tnsIndex i = 0; i < length; ++i)
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
            for(int mode_i = spatsrX->nmodes - 1; mode_i >= 0; --mode_i){
                if(mode_i == copt_mode)
                    spatsrY->inds[mode_i].values[nnz_j] = 0;
                else{
                    spatsrY->inds[mode_i].values[nnz_j]  = indexs % spatsrY->ndims[mode_i];
                    indexs = indexs / spatsrY->ndims[mode_i];
                }///< else
            }///< for mode_i
            ++nnz_j;
        }/// <if
    }///< for nnz_i
    tnsFreeValueVector(&datavec);
    return 0;
}


///  并行Sparse tensor times a vector (SpTTV)
int tnsOmpTTVSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsValueVector *vec, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= spatsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(spatsrX->ndims[copt_mode] != vec->nlens || spatsrY->ndims[copt_mode] != 1){
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

    // 获取spatsrY的松散上界并创建Valuevacter datavec存储中间值
    for(tnsIndex i = 0; i < spatsrX->nmodes; ++i){
        if(i != copt_mode)
            length  = length * spatsrX->ndims[i];
    } 

    tnsValueVector datavec;
    tnsNewValueVector(&datavec, length);

    // 计算values并统计spatsrY的非零元数
    #pragma omp parallel for num_threads(nthreads)
    for(tnsIndex nnz_i = 0; nnz_i < spatsrX->nnz; ++nnz_i){
        data_a = spatsrX->values.values[nnz_i];
        data_b = vec->values[spatsrX->inds[copt_mode].values[nnz_i]];
        hash_index = 0;
        hash_step = 1;
        // 计算spatsrY的values以及每个值对应的位置hash_index
        for(int mode_i = spatsrX->nmodes - 1; mode_i >= 0; --mode_i){
            if(mode_i != copt_mode){
                hash_index += spatsrX->inds[mode_i].values[nnz_i] * hash_step;
                hash_step *= spatsrX->ndims[mode_i];
            }///< if
        }///< for mode_i
        datavec.values[hash_index] += data_a * data_b;
    }///< for nnz_i
    
    for(tnsIndex i = 0; i < length; ++i)
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
    for(tnsIndex nnz_i = 0; nnz_i < length; ++nnz_i){
        if(datavec.values[nnz_i] != 0){
            spatsrY->values.values[nnz_j] = datavec.values[nnz_i];
            indexs = nnz_i;
            for(int mode_i = spatsrX->nmodes-1; mode_i >= 0; --mode_i){
                if(mode_i == copt_mode)
                    spatsrY->inds[mode_i].values[nnz_j] = 0;
                else{
                    spatsrY->inds[mode_i].values[nnz_j]  = indexs % spatsrY->ndims[mode_i];
                    indexs = indexs / spatsrY->ndims[mode_i];
                }///< else
            }///< for mode_i
            ++nnz_j;
        }/// <if
    }///< for nnz_i
    tnsFreeValueVector(&datavec);
    return 0;
}


/*
* 稠密张量
*/
///  稠密张量TTV
int tnsTTVDentsr(tnsDenseTensor *dentsrY, const tnsDenseTensor *dentsrX, const tnsValueVector *vec, tnsIndex copt_mode) {
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= dentsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(dentsrX->ndims[copt_mode] != vec->nlens || dentsrY->ndims[copt_mode] != 1){
        printf("维度不匹配\n");
        return 1;
    }
    // Timer t;
    // timer_start(&t);

    // tnsIndex row = 1;
    // tnsIndex col = dentsrX->ndims[copt_mode];
    // for(tnsIndex i = 0; i < dentsrX->nmodes; ++i)
    //     if(i != copt_mode)
    //         row *= dentsrX->ndims[i];
    
    // tnsDenseMatrix mtx;
    // tnsNewDenseMatrix(&mtx, row, col);
    // for(tnsIndex i = 0; i < dentsrX->nnz; ++i)
    //     mtx.values.values[i] = dentsrX->values.values[i];
    // tnsValueVector imdvec;
    // tnsNewValueVector(&imdvec, row);

    // for(tnsIndex i = 0; i < mtx.nrows; ++i)
    //     for(tnsIndex j = 0; j < mtx.ncols; ++j)
    //         imdvec.values[i] += mtx.values.values[i * mtx.stride + j] * vec->values[j];
    // timer_stop(&t);
    // timer_print_sec(&t, "mtx");
    // timer_restart(&t);
    // for(tnsIndex i = 0; i < imdvec.nlens; ++i)
    //     dentsrY->values.values[i] = imdvec.values[i];
    // timer_stop(&t);
    // timer_print_sec(&t, "move");
    
    // for(tnsIndex i = 0; i < imdvec.nlens; ++i)
    //    printf("%f ", imdvec.values[i]);
    // printf("\n");

    // timer_restart(&t);
    tnsValue ele_a;
    tnsValue ele_b;
    tnsIndex hash_indexY;
    tnsIndex step = 1;
    tnsIndex hash_indexV;
    for(tnsIndex i = 0; i < dentsrX->nmodes; ++i)
        if(i != copt_mode)
            step *= dentsrX->ndims[i];

    for(tnsIndex nnz_i = 0; nnz_i < dentsrX->nnz; ++nnz_i){
    	hash_indexY = nnz_i / dentsrX->ndims[copt_mode];
        hash_indexV = nnz_i % dentsrX->ndims[copt_mode];

        ele_a = dentsrX->values.values[nnz_i];
        ele_b = vec->values[hash_indexV];

        dentsrY->values.values[hash_indexY] += ele_a * ele_b;
    }
    // timer_stop(&t);
    // timer_print_sec(&t, "tv");

    // for(tnsIndex i = 0; i < dentsrY->nnz; ++i)
    //    printf("%f ", dentsrY->values.values[i]);
    // printf("\n");

    return 0;
}

///  并行稠密张量TTV
int tnsOmpTTVDentsr(tnsDenseTensor *dentsrY, const tnsDenseTensor *dentsrX, const tnsValueVector *vec, tnsIndex copt_mode) {
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= dentsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(dentsrX->ndims[copt_mode] != vec->nlens || dentsrY->ndims[copt_mode] != 1){
        printf("维度不匹配\n");
        return 1;
    }

    tnsValue ele_a;
    tnsValue ele_b;
    tnsIndex hash_indexY;
    tnsIndex step = 1;
    tnsIndex hash_indexV;
    for(tnsIndex i = 0; i < dentsrX->nmodes; ++i)
        if(i != copt_mode)
            step *= dentsrX->ndims[i];

    #pragma omp parallel for num_threads(nthreads)
    for(tnsIndex nnz_i = 0; nnz_i < dentsrX->nnz; ++nnz_i){
        hash_indexY = nnz_i / dentsrX->ndims[copt_mode];
        hash_indexV = nnz_i / step;

        ele_a = dentsrX->values.values[nnz_i];
        ele_b = vec->values[hash_indexV];

        dentsrY->values.values[hash_indexY] += ele_a * ele_b;
    }

    return 0;
}


