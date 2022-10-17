#include <TArm.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>

// Y的维度和非零元数量已知；计算前必须先重序,copt_mode一定是最后一个mode
int tnsVecTilingTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr){
    // 前提是非零元已经按维度大小顺序排好，copt_mode放在了最后面，其他mode相同的元素已经放在一起了；
    // 首先进行划分tile，也就是统计结果。
    // tnsIndexVector one_tile_vec;
    tnsFreeIndexVector(&X_tsr->tile_ptr_begin);
    tnsNewIndexVector(&X_tsr->tile_ptr_begin, 0);
    tnsFreeIndexVector(&X_tsr->tile_ptr_end);
    tnsNewIndexVector(&X_tsr->tile_ptr_end, 0);
    // 追加tile起始位置
    tnsAppendIndexVector(&X_tsr->tile_ptr_begin, 0);
    // 通过比较上下两个元素的除了copt_mode外的其他mode是否相同得到是否append元素。
    for(tnsIndex nnz_i = 1; nnz_i < X_tsr->nnz; ++nnz_i){
        // 不比较最后一个维度，因为最后一个是计算维度
        for(int mode_i = X_tsr->nmodes - 2; mode_i >= 0; --mode_i){
            if(X_tsr->inds[mode_i].values[nnz_i] != X_tsr->inds[mode_i].values[nnz_i-1]){
                tnsAppendIndexVector(&X_tsr->tile_ptr_begin, nnz_i);
                tnsAppendIndexVector(&X_tsr->tile_ptr_end, nnz_i);
                break;
            }///< if
        }///< for mode_i
    }
    // 追加tile最后位置（类似于CSR的ptr，要比tile的数目多1）
    tnsAppendIndexVector(&X_tsr->tile_ptr_end, X_tsr->nnz);

    // 根据 one_tile_vec 的大小初始化Y的非零元对应的向量大小
    for(tnsIndex mode = 0; mode < Y_tsr->nmodes; ++mode){
        tnsFreeIndexVector(&Y_tsr->inds[mode]);
        tnsNewIndexVector(&Y_tsr->inds[mode], X_tsr->tile_ptr_end.nlens);
    }
    // printf("tile_num %u\n", X_tsr->tile_ptr_end.nlens);
    tnsFreeIndexVector(&Y_tsr->values);
    tnsNewValueVector(&Y_tsr->values, X_tsr->tile_ptr_end.nlens);
    Y_tsr->nnz = X_tsr->tile_ptr_end.nlens;

    // 提前给TTV的坐标赋值
    #pragma omp parallel for num_threads(32)
    for(tnsIndex tile_i = 0; tile_i < X_tsr->tile_ptr_end.nlens; ++tile_i){
        
        // 给Y的每个非零元赋值模态
        for(tnsIndex mode_i = 0; mode_i < X_tsr->nmodes-1; ++mode_i){
            Y_tsr->inds[mode_i].values[X_tsr->tile_ptr_begin.values[tile_i]] = X_tsr->inds[mode_i].values[X_tsr->tile_ptr_begin.values[tile_i]];
        }
    }


	return 0;
}

// 统计X在mode下非零元的不同索引的数量，得到的就是Y的非零元个数。
// 输出是tnsSparseTensor ，输入是tnsTileSpatsr；Y的维度和非零元数量已知；计算前必须先重序,copt_mode一定是最后一个mode
int tnsTTVTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr, tnsValueVector *vec, tnsIndex const copt_mode, const tnsIndex tk){

    // 根据 one_tile_vec 计算 TTV
    // #pragma omp parallel for num_threads(tk)
    for(tnsIndex tile_i = 0; tile_i < X_tsr->tile_ptr_end.nlens; tile_i++){
        // tnsValue sum = 0;
        // 遍历一个tile中的非零元，累加结果
        for(tnsIndex nnz_i = X_tsr->tile_ptr_begin.values[tile_i]; nnz_i < X_tsr->tile_ptr_end.values[tile_i]; ++nnz_i){
            // #pragma omp atomic update
            Y_tsr->values.values[tile_i] += X_tsr->values.values[nnz_i] * vec->values[X_tsr->inds[copt_mode].values[nnz_i]];
        }
        // 赋值给Y
        // Y_tsr->values.values[tile_i] = sum;
    }

	return 0;
}


// 统计X在mode下非零元的不同索引的数量，得到的就是Y的非零元个数。
// 输出是tnsSparseTensor ，输入是tnsTileSpatsr；Y的维度和非零元数量已知；计算前必须先重序,copt_mode一定是最后一个mode
int tnsOMPTTVTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr, tnsValueVector *vec, tnsIndex const copt_mode, const tnsIndex tk){
    // // 前提是非零元已经按维度大小顺序排好，copt_mode放在了最后面，其他mode相同的元素已经放在一起了；
    // // 首先进行划分tile，也就是统计结果。
    // tnsIndexVector one_tile_vec;
    // tnsNewIndexVector(&one_tile_vec, 0);
    // // 追加tile起始位置
    // tnsAppendIndexVector(&one_tile_vec, 0);
    // // 通过比较上下两个元素的除了copt_mode外的其他mode是否相同得到是否append元素。
    // for(tnsIndex nnz_i = 1; nnz_i < X_tsr->nnz; ++nnz_i){
    //     // 不比较最后一个维度，因为最后一个是计算维度
    //     for(int mode_i = X_tsr->nmodes - 2; mode_i >= 0; --mode_i){
    //         if(X_tsr->inds[mode_i].values[nnz_i] != X_tsr->inds[mode_i].values[nnz_i-1]){
    //             tnsAppendIndexVector(&one_tile_vec, nnz_i);
    //             break;
    //         }///< if
    //     }///< for mode_i
    // }
    // // 追加tile最后位置（类似于CSR的ptr，要比tile的数目多1）
    // tnsAppendIndexVector(&one_tile_vec, X_tsr->nnz);
    // // 根据 one_tile_vec 的大小初始化Y的非零元对应的向量大小
    // for(tnsIndex mode = 0; mode < Y_tsr->nmodes; ++mode){
    //     tnsFreeIndexVector(&Y_tsr->inds[mode]);
    //     tnsNewIndexVector(&Y_tsr->inds[mode], one_tile_vec.nlens-1);
    // }
    // tnsFreeIndexVector(&Y_tsr->values);
    // tnsNewValueVector(&Y_tsr->values, one_tile_vec.nlens-1);

    // 根据 one_tile_vec 计算 TTV
    #pragma omp parallel for num_threads(tk)
    for(tnsIndex tile_i = 0; tile_i < X_tsr->tile_ptr_end.nlens; ++tile_i){
        // tnsValue sum = 0;
        // 遍历一个tile中的非零元，累加结果
        for(tnsIndex nnz_i = X_tsr->tile_ptr_begin.values[tile_i]; nnz_i < X_tsr->tile_ptr_end.values[tile_i]; ++nnz_i){
            // #pragma omp atomic update
            Y_tsr->values.values[tile_i] += X_tsr->values.values[nnz_i] * vec->values[X_tsr->inds[copt_mode].values[nnz_i]];
        }
        // 赋值给Y
        // Y_tsr->values.values[tile_i] = sum;
    }

	return 0;
}
