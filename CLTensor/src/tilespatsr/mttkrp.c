#include <TArm.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>

// 不分tile直接计算MTTKRP
int tnsNaiveMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const copt_mode, const tnsIndex tk){
    tnsIndex rank = U_list[0]->ncols;
    // 结果矩阵
    tnsValue *res_array = calloc(tsr->nnz_ndims[copt_mode]*rank, sizeof *res_array);
    // tnsValue *res_array = calloc(tsr->ndims[copt_mode]*rank, sizeof *res_array);
    // 需要额外的储存空间
    // tnsIndex stride = mats[0]->stride;
    
    
    // printf("start omp computer %d\n", tsr->nnz_ndims[copt_mode]);
    // 遍历每个非零元得到计算结果
    #pragma omp parallel for num_threads(tk)
    // #pragma omp parallel for num_threads(tk) reduction(+:res_array[0:tsr->nnz_ndims[copt_mode]*rank])
    for(tnsIndex nnz_i = 0; nnz_i < tsr->nnz; ++nnz_i){
        tnsValue val = tsr->values.values[nnz_i];
        tnsValueVector scratch;  // Temporary array
        tnsNewValueVector(&scratch, rank);
        tnsConstantValueVector(&scratch, 0);
        #pragma omp simd
        for(tnsIndex r=0; r<rank; ++r) {
            scratch.values[r] = val;
        }
        // 先算mode，内部再用simd
        for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
            if(mode_i == copt_mode)continue;
            #pragma omp simd
            for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
                val *= U_list[mode_i]->values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i]->stride + rank_i];
            }
            
        }
        for(tnsIndex rank_i=0; rank_i<rank; ++rank_i) {
            #pragma omp atomic update
            res_array[tsr->inds[copt_mode].values[nnz_i] * rank + rank_i] += scratch.values[rank_i];
        }
        // for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
        //     for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
        //         if(mode_i == copt_mode)continue;
        //         val *= U_list[mode_i]->values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i]->stride + rank_i];
        //     }
        //     // 结果保存到矩阵的相应位置
        //     res_array[tsr->inds[copt_mode].values[nnz_i] * rank + rank_i] += val;
            
        // } ///< for rank_i
        
    } ///< for nnz_i
    // printf("computer success!\n");
    // 将数组值还原到U_list的临时矩阵中
    for(tnsIndex i = 0; i < tsr->nnz_ndims[copt_mode]*rank; ++i){
        U_list[tsr->nmodes]->values.values[(i/rank) * U_list[copt_mode]->stride + (i%rank)] = res_array[i];
    }
    
    // for(tnsIndex i = 0; i < U_list[copt_mode].nrows; ++i){
    //     for(tnsIndex j = 0; j < U_list[copt_mode].ncols; j++){
    //         U_list[copt_mode].values.values[i * U_list[copt_mode].stride + j] = res_array[i * rank + j];
    //     }
    // }
    // for(int i = 0; i < tsr->nnz_ndims[tsr->layer_dim]*rank; ++i){
    //     printf("%f ", res_array[i]);
    // }
    free(res_array);
    // printf("succesee! \n");
    // tnsFreeDenseMatrix(&U_list[tsr->layer_dim]);
    // U_list[tsr->layer_dim] = res_mat;


	return 0;
}

int tnsOMPNaiveMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const copt_mode, const tnsIndex tk){
    tnsIndex rank = U_list[0]->ncols;
    // 结果矩阵
    tnsValue *res_array = calloc(tsr->nnz_ndims[copt_mode]*rank, sizeof *res_array);
    // tnsValue *res_array = calloc(tsr->ndims[copt_mode]*rank, sizeof *res_array);
    // 需要额外的储存空间
    // tnsIndex stride = mats[0]->stride;
    
    
    // printf("start omp computer %d\n", tsr->nnz_ndims[copt_mode]);
    // 遍历每个非零元得到计算结果
    // #pragma omp parallel for num_threads(tk)
    #pragma omp parallel for num_threads(tk) reduction(+:res_array[0:tsr->nnz_ndims[copt_mode]*rank])
    for(tnsIndex nnz_i = 0; nnz_i < tsr->nnz; ++nnz_i){
        tnsValue val = tsr->values.values[nnz_i];
        tnsValueVector scratch;  // Temporary array
        tnsNewValueVector(&scratch, rank);
        tnsConstantValueVector(&scratch, 0);
        #pragma omp simd
        for(tnsIndex r=0; r<rank; ++r) {
            scratch.values[r] = val;
        }
        // 先算mode，内部再用simd
        for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
            if(mode_i == copt_mode)continue;
            #pragma omp simd
            for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
                val *= U_list[mode_i]->values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i]->stride + rank_i];
            }
            
        }
        for(tnsIndex rank_i=0; rank_i<rank; ++rank_i) {
            // #pragma omp atomic update
            res_array[tsr->inds[copt_mode].values[nnz_i] * rank + rank_i] += scratch.values[rank_i];
        }


        // for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
        //     for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
        //         if(mode_i == copt_mode)continue;
        //         val *= U_list[mode_i]->values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i]->stride + rank_i];
        //     }
        //     // 结果保存到矩阵的相应位置
        //     res_array[tsr->inds[copt_mode].values[nnz_i] * rank + rank_i] += val;
            
        // } ///< for rank_i
        
    } ///< for nnz_i
    // printf("computer success!\n");
    // 将数组值还原到U_list的临时矩阵中
    for(tnsIndex i = 0; i < tsr->nnz_ndims[copt_mode]*rank; ++i){
        U_list[tsr->nmodes]->values.values[(i/rank) * U_list[copt_mode]->stride + (i%rank)] = res_array[i];
    }
    
    // for(tnsIndex i = 0; i < U_list[copt_mode].nrows; ++i){
    //     for(tnsIndex j = 0; j < U_list[copt_mode].ncols; j++){
    //         U_list[copt_mode].values.values[i * U_list[copt_mode].stride + j] = res_array[i * rank + j];
    //     }
    // }
    // for(int i = 0; i < tsr->nnz_ndims[tsr->layer_dim]*rank; ++i){
    //     printf("%f ", res_array[i]);
    // }
    free(res_array);
    // printf("succesee! \n");
    // tnsFreeDenseMatrix(&U_list[tsr->layer_dim]);
    // U_list[tsr->layer_dim] = res_mat;


	return 0;
}

/// 分tile计算MTTKRP mode == tsr->layer_dim
int tnsOmpMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const copt_mode, const tnsIndex tk){
    tnsIndex rank = U_list[0]->ncols;
    // 结果矩阵
    // tnsDenseMatrix res_mat;
    // tnsNewDenseMatrix(&res_mat, tsr->ndims[tsr->layer_dim], rank);
    tnsValue *res_array = calloc(tsr->nnz_ndims[tsr->layer_dim]*rank, sizeof *res_array);

    // for(int i = 0; i < tsr->nnz_ndims[tsr->layer_dim]*rank; ++i){
    //     printf("%f ", res_array[i]);
    // }
    // 遍历每一层
    // #pragma omp parallel for num_threads(tk)
    for(tnsIndex lay_i = 0; lay_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++lay_i){
        // 对应A矩阵的位置
        tnsIndex res_begin = 0;
        tnsIndex res_end = 0;
        res_begin = lay_i * tsr->ndim_blocks[tsr->layer_dim] * rank;
        // 不是最后一个layer时
        if(lay_i < tsr->ndim_count_blocks.values[tsr->layer_dim] - 1){
            res_end = (lay_i+1) * tsr->ndim_blocks[tsr->layer_dim] * rank;
        }else{
            res_end = tsr->nnz_ndims[tsr->layer_dim] * rank;
        }
        // printf("res_begin :%d, %d\n",tsr->layer_ptr.values[lay_i], tsr->layer_ptr.values[lay_i + 1]);

        // 遍历layer中所有tile
        #pragma omp parallel for num_threads(tk) reduction(+:res_array[res_begin:res_end])
        for(tnsIndex tile_i = 0; tile_i < tsr->layer_tiles[lay_i].nlens; ++tile_i){
            // printf("res_begin :%d, %d\n",tsr->tile_ptr_begin.values[tsr->layer_tiles[lay_i].values[tile_i]], tsr->tile_ptr_end.values[tsr->layer_tiles[lay_i].values[tile_i]]);

            // 遍历tile中所有非零元
            for(tnsIndex nnz_i = tsr->tile_ptr_begin.values[tsr->layer_tiles[lay_i].values[tile_i]]; nnz_i < tsr->tile_ptr_end.values[tsr->layer_tiles[lay_i].values[tile_i]]; ++nnz_i){
                tnsValue val = tsr->values.values[nnz_i];
                for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
                    for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
                        if(mode_i == tsr->layer_dim)continue;
                        val *= U_list[mode_i]->values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i]->stride + rank_i];
                    }
                    // 结果保存到矩阵的相应位置
                    res_array[tsr->inds[tsr->layer_dim].values[nnz_i] * rank + rank_i] += val;
                    
                } ///< for rank_i

            } ///< for nnz_i

        } ///< for tile_i

    } ///< for lay_i

    // 将数组值还原到矩阵中
    // tnsIndex i,j;
    // #pragma omp parallel num_threads(tk) private(i,j)
    // {
    // #pragma omp for schedule(dynamic)
    // for(i = 0; i < U_list[tsr->layer_dim].nrows; ++i){
    //     for(j = 0; j < U_list[tsr->layer_dim].ncols; j++){
    //         U_list[tsr->layer_dim].values.values[i * U_list[tsr->layer_dim].stride + j] = res_array[i * rank + j];
    //     }
    // }
    // }
    // for(int i = 0; i < tsr->nnz_ndims[tsr->layer_dim]*rank; ++i){
    //     printf("%f ", res_array[i]);
    // }
    // 将数组值还原到矩阵中。
    for(tnsIndex i = 0; i < tsr->nnz_ndims[copt_mode]*rank; ++i){
        U_list[tsr->nmodes]->values.values[(i/rank) * U_list[copt_mode]->stride + (i%rank)] = res_array[i];
    }

    // for(tnsIndex i = 0; i < U_list[mode].nrows; ++i){
    //     for(tnsIndex j = 0; j < U_list[mode].ncols; j++){
    //         U_list[mode].values.values[i * U_list[mode].stride + j] = res_array[i * rank + j];
    //     }
    // }
    free(res_array);

	return 0;
}



// /// 分tile计算MTTKRP
// int tnsMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix *U_list){
//     tnsIndex rank = U_list[0].ncols;
//     // 结果矩阵
//     tnsDenseMatrix res_mat;
//     tnsNewDenseMatrix(&res_mat, tsr->ndims[tsr->layer_dim], rank);

//     // 遍历每一层
//     for(tnsIndex lay_i = 0; lay_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++lay_i){

//         // 遍历layer中所有tile
//         for(tnsIndex tile_i = tsr->layer_ptr.values[lay_i]; tile_i < tsr->layer_ptr.values[lay_i + 1]; ++tile_i){

//             // 遍历tile中所有非零元
//             for(tnsIndex nnz_i = tsr->tile_ptr_begin.values[tile_i]; nnz_i < tsr->tile_ptr_end.values[tile_i]; ++nnz_i){
//                 tnsValue val = tsr->values.values[nnz_i];
//                 // 找到各个模态对应的矩阵的位置，逐元素乘加
//                 for(tnsIndex rank_i = 0; rank_i < rank; ++rank_i){
//                     for(tnsIndex mode_i = 0; mode_i < tsr->nmodes; ++mode_i){
//                         if(mode_i == tsr->layer_dim)continue;
//                         // 乘
//                         val *= U_list[mode_i].values.values[tsr->inds[mode_i].values[nnz_i] * U_list[mode_i].stride + rank_i];
//                     }
//                     // 结果保存到矩阵的相应位置
//                     // 加
//                     res_mat.values.values[tsr->inds[tsr->layer_dim].values[nnz_i] * res_mat.stride + rank_i] += val;
                
//                 } ///< for rank_i
                
//             } ///< for nnz_i

//         } ///< for tile_i

//     } ///< for lay_i
//     tnsFreeDenseMatrix(&U_list[tsr->layer_dim]);
//     U_list[tsr->layer_dim] = res_mat;
// 	return 0;
// }