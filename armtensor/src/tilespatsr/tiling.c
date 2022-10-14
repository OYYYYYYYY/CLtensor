#include <TArm.h>
#include <stdlib.h>
#include <string.h>

static int CMP_QuickSort(tnsTileSpatsr *A, tnsIndex i, tnsIndex* key_index){
    for(tnsIndex k = 0; k < A->nmodes; ++k){
        if(A->inds[k].values[i] == key_index[k]) continue;
        return A->inds[k].values[i] > key_index[k];
    }
   
}

/// multidimensional Valuation, ith element <- jth element
static int Mult_Valuation(tnsTileSpatsr *A, tnsIndex i, tnsIndex j){
    tns_CheckError(i > A->nnz || j > A->nnz, "Mult_Valuation", "超出张量非零元的范围");
    for(tnsIndex k = 0; k < A->nmodes; ++k)
        A->inds[k].values[i]=A->inds[k].values[j];
    A->values.values[i] = A->values.values[j];
    return 0;
}

static int tnsQuickSortSpatsr(tnsTileSpatsr *A, int begin, int end){
    if(begin < end){
        tnsIndex *key_index = calloc(A->nmodes,sizeof(tnsIndex));
        tnsValue key_val;
        //SaveNnz(A, key_index, begin, &key_val);
        for(tnsIndex i=0;i<A->nmodes;++i) key_index[i]=A->inds[i].values[begin];
        key_val = A->values.values[begin];
        tnsIndex i = begin;
        tnsIndex j = end;
        while(i < j){
            while(i < j && CMP_QuickSort(A, j, key_index)){
                --j;
            }
            if(i < j){
                Mult_Valuation(A, i, j);
                ++i;
            }
            while(i < j && !CMP_QuickSort(A, i, key_index)){
                ++i;
            }
            if(i < j){
                Mult_Valuation(A, j, i);
                --j;
            }
        }
        for(tnsIndex k=0;k<A->nmodes;++k) A->inds[k].values[i]=key_index[k];
        A->values.values[i]=key_val;
        key_val=A->values.values[begin];        
        tnsQuickSortSpatsr(A, begin, i-1);
        tnsQuickSortSpatsr(A, i+1, end);
        free(key_index);
    }
    return 0;
}

// 非递归的快排
static int tnsNRQuickSortSpatsr(tnsTileSpatsr *A, int n){
    int i, j, begin, end, temp;
	int top = 0;
    // stackNode *st = malloc(sizeof(stackNode)*A->nnz);
    int *st_low = malloc(sizeof(int)*A->nnz);
    int *st_high = malloc(sizeof(int)*A->nnz);

    st_low[top] = 0;
	st_high[top] = n-1;
    while( top > -1){
		begin = st_low[top];
		end = st_high[top];
		top--;
		i = begin;
		j = end;
 
		if( begin < end ){
			tnsIndex *key_index = calloc(A->nmodes,sizeof(tnsIndex));
            tnsValue key_val;
            for(tnsIndex i=0;i<A->nmodes;++i) key_index[i]=A->inds[i].values[begin];
            key_val = A->values.values[begin];
            // tnsIndex i = begin;
            // tnsIndex j = end;
			while( i < j){
                while(i < j && CMP_QuickSort(A, j, key_index)){
                    --j;
                }
                Mult_Valuation(A, i, j);

                while(i < j && !CMP_QuickSort(A, i, key_index)){
                    ++i;
                }
                Mult_Valuation(A, j, i); 
            }
 
            for(tnsIndex k=0;k<A->nmodes;++k) A->inds[k].values[i]=key_index[k];
            A->values.values[i] = key_val;
            key_val=A->values.values[begin]; 

			if(i <= j){
				top++;
				st_low[top] = begin;
				st_high[top] = i-1;
			
				top++;
				st_low[top] = ++i;
				st_high[top] = end;
			}
            free(key_index);
		}
	}
    free(st_low);
    free(st_high);



    return 0;
}
/**
  * @fn tnsPermuteTileSpatsr
  * @brief 对张量根据dim顺序从小到大进行排序。
  * @details 根据 ndims 的顺序，按dims内部从小到大递增的重新对nnz排序。
  * @param[in out] tsr 返回有顺序的张量
  * @return 返回转置和排序后的张量
  */
int tnsPermuteTileSpatsr(tnsTileSpatsr *tsr, tnsIndex *order){
    tnsIndexVector *temp_inds = malloc(tsr->nmodes * sizeof *temp_inds);
    memcpy(temp_inds, tsr->inds, tsr->nmodes * sizeof *temp_inds);

    tnsIndex *temp_ndims = malloc(tsr->nmodes * sizeof *temp_ndims);
    memcpy(temp_ndims, tsr->ndims, tsr->nmodes * sizeof *temp_ndims);

	tnsIndexVector *temp_alldim_nnzs = malloc(tsr->nmodes * sizeof *temp_alldim_nnzs);
    memcpy(temp_alldim_nnzs, tsr->alldim_nnzs, tsr->nmodes * sizeof *temp_alldim_nnzs);
    //permute
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        tsr->ndims[i] = temp_ndims[order[i]];
        tsr->inds[i] = temp_inds[order[i]];
        tsr->alldim_nnzs[i] = temp_alldim_nnzs[order[i]];
    }
    //sort
    tnsNRQuickSortSpatsr(tsr, tsr->nnz);
    // tnsQuickSortSpatsr(tsr, 0, tsr->nnz-1);
    free(temp_alldim_nnzs);
    free(temp_inds);
    free(temp_ndims);
    return 0;
}



static int tnsPartitionIndexVec(tnsIndexVector *raw_vec, tnsIndexVector *idx_vec, int begin, int end){
    int key = begin;
    while(begin < end){
        while(key < end){
            //注意处理等值的情况，这里将等值置于大堆
            if(raw_vec->values[key] >= raw_vec->values[end]){
                end--;
            }else{
                // swap key 和 end 上的位置
                tnsIndex tmp = raw_vec->values[end];
                raw_vec->values[end] = raw_vec->values[key];
                raw_vec->values[key] = tmp;

                // swap keys when swap values
                tmp = idx_vec->values[end];
                idx_vec->values[end] = idx_vec->values[key];
                idx_vec->values[key] = tmp;

                key = end;
            }
        }
        while(key > begin){
            
            if(raw_vec->values[key] < raw_vec->values[begin]){
                begin++;
            }else{
                // swap key 和 end 上的位置
                tnsIndex tmp = raw_vec->values[begin];
                raw_vec->values[begin] = raw_vec->values[key];
                raw_vec->values[key] = tmp;

                // swap keys when swap values
                tmp = idx_vec->values[begin];
                idx_vec->values[begin] = idx_vec->values[key];
                idx_vec->values[key] = tmp;

                key = begin;
            }
        }
    }
    return key;
}



/**
  * @fn tnsQuitSortIndexVector
  * @brief 对向量根据dim顺序从大到小进行排序。
  * @details 对元素快排，并返回新数组上的顺序元素 在 原数组中的位置 的数组。
  * @param[in out] raw_vec 输入原数组，并返回有顺序的数组
  * @param[in out] idx_vec 输入原数组的index，并返回新的数组中在原数组中对应的值（可以是下标）
  * @param[in] begin 开始的index，初始为0.
  * @param[in] end 结束的index，初始为len-1.
  * @return 返回转置和排序后的张量
  */
static int tnsQuitSortIndexVector(tnsIndexVector *raw_vec, tnsIndexVector *idx_vec, int begin, int end){
    if(begin < end){
        //这里完成一趟排序，将arr[k]置于正确位置
        int key = tnsPartitionIndexVec(raw_vec, idx_vec, begin, end);
        // k = sort(i,j,arr);
        //这里处理arr[k]左侧区间
        if(key > begin){
            tnsQuitSortIndexVector(raw_vec, idx_vec, begin, key-1);
        }
        //这里处理arr[k]右侧区间
        if(key < end){
            tnsQuitSortIndexVector(raw_vec, idx_vec, key+1, end);
        }
    }

    return 0;
}

// 非递归的快排实现
static int tnsNRQuitSortIndexVector(tnsIndexVector *raw_vec, tnsIndexVector *idx_vec, int n){
    int i, j, begin, end;
    tnsIndex temp_val, tenmp_idx;
	int top = 0;
    // stackNode *st = malloc(sizeof(stackNode)*A->nnz);
    int *st_low = malloc(sizeof(int)*n);
    int *st_high = malloc(sizeof(int)*n);

    st_low[top] = 0;
	st_high[top] = n-1;
    while( top > -1){
		begin = st_low[top];
		end = st_high[top];
		top--;
		i = begin;
		j = end;
 
		if( begin < end ){

			temp_val = raw_vec->values[begin];
			tenmp_idx = idx_vec->values[begin];
            while( i < j){
                while(i < j && raw_vec->values[j] >= temp_val){
                    --j;
                }
                // swap key 和 end 上的位置
                raw_vec->values[i] = raw_vec->values[j];
                idx_vec->values[i] = idx_vec->values[j];

                while(i < j && raw_vec->values[j] < temp_val){
                    ++i;
                }
                raw_vec->values[j] = raw_vec->values[i];
                idx_vec->values[j] = idx_vec->values[i];
            }
			
			raw_vec->values[i] = temp_val;
			idx_vec->values[i] = tenmp_idx;
            
 
			if(i <= j){
				top++;
				st_low[top] = begin;
				st_high[top] = i-1;
			
				top++;
				st_low[top] = ++i;
				st_high[top] = end;
			}
            
		}
	}
    free(st_low);
    free(st_high);

    return 0;
}


/**
  * @fn tnsRedimTileSpatsr
  * @brief dim重序，再根据新的dim对nnz重序, 去除全0dim，产生nnz_ndims
  * @details 目的是：密集dim的在前面，稀疏dim的在后面。nnz的坐标改变需要映射，alldim_raw_index记录映射关系 X=>Y 索引与X一致，值是X上的该元素在Y的位置。根据映射，修改 inds ， alldim_nnzs
  * @param[in out] tsr 输入原张量，并返回有顺序的改变后的张量
  * @return 返回前稠密后稀疏的排序后的张量
  */
int tnsRedimTileSpatsr(tnsTileSpatsr *tsr){
    // 原数组的位置上的元素 在 新数组上的新位置。
    for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){

        // 只计算非零维度的坐标
        tnsIndexVector temp_alldim_nnzs;
        tnsIndexVector temp_nnzs_idx;
        tnsNewIndexVector(&temp_alldim_nnzs, 0);
        tnsNewIndexVector(&temp_nnzs_idx, 0);
        // 根据非零元的分布，去除掉全0的dim，得到新的维度数。
        tnsIndex i;
        // #pragma omp parallel for num_threads(32)  private(i) reduction(+:zero_dim_num)
        for(i = 0; i < tsr->ndims[mode]; ++i){
            if(tsr->alldim_nnzs[mode].values[i] != 0){
                tnsAppendIndexVector(&temp_alldim_nnzs, tsr->alldim_nnzs[mode].values[i]);
                tnsAppendIndexVector(&temp_nnzs_idx, i);
            }
        }
        // printf("success temp_alldim_nnzs\n");
        tsr->nnz_ndims[mode] = temp_alldim_nnzs.nlens;
        // tnsNewIndexVector(&temp_nnzs_idx, tsr->ndims[mode]);
        // tnsContinuousIndexVector(&temp_nnzs_idx, 0, 1);
        // printf("success %d \n", temp_alldim_nnzs.nlens);
        // 10万非零元可以，20万就会报段错误。（可以改成非递归类型的快排）
        tnsNRQuitSortIndexVector(&temp_alldim_nnzs, &temp_nnzs_idx, temp_alldim_nnzs.nlens);
        // tnsQuitSortIndexVector(&temp_alldim_nnzs, &temp_nnzs_idx, 0, temp_alldim_nnzs.nlens-1);
        // printf("success tnsQuitSortIndexVector\n");
        // 排序完后，重新赋值的tsr->alldim_nnzs，确保维度随之改变
        tnsConstantIndexVector(&tsr->alldim_nnzs[mode], 0);
        tnsPartCopyIndexVector(&tsr->alldim_nnzs[mode], &temp_alldim_nnzs, temp_alldim_nnzs.nlens);
        

        // 为了方便映射，将返回的新数组顺序，调整为原数组顺序，即key和value交换。
        // printf("success tnsPartCopyIndexVector\n");
        tnsMapSwapIndexVector(&tsr->alldim_raw_indexs[mode], &temp_nnzs_idx);
        tnsFreeIndexVector(&temp_nnzs_idx);
        tnsFreeIndexVector(&temp_alldim_nnzs);
        // 根据该模态的map， 更改每个nnz的inds[mode]
        for(tnsIndex i = 0; i < tsr->nnz; ++i){
            tsr->inds[mode].values[i] = tsr->alldim_raw_indexs[mode].values[tsr->inds[mode].values[i]];
        }

        // printf("mode success!\n");
    }
    // inds调整完后，需要重新排序
    tnsIndexVector new_order_vec;
    tnsNewIndexVector(&new_order_vec, tsr->nmodes);
    tnsContinuousIndexVector(&new_order_vec, 0, 1);
    tnsPermuteTileSpatsr(tsr, new_order_vec.values);
    tnsFreeIndexVector(&new_order_vec);
	return 0;
}

/**
  * @fn tnsTilingTileSpatsr
  * @brief 对nnz按分块顺序存储。
  * @details 根据cache的大小，划分成tile，并根据tile顺序对nnz排序，ndims的顺序是tile之间的顺序。
  * @param[in out] tsr 输入原张量，并返回有顺序的改变后的张量
  * @param[in] ndim_block_sizes 每个维度的分块大小
  * @param[in] sp_threshold     维度的稀疏阈值
  * @return 返回前稠密后稀疏的排序后的张量
  */
int tnsTilingTileSpatsr(tnsTileSpatsr *tsr, tnsIndex *ndim_block_sizes, tnsIndex sp_threshold){
    // 根据分块的大小生成对应数量的tile
    tnsIndex spa_dim_num = 0;
    tsr->tiles_num = 1;
    for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){
        spa_dim_num = 0;
        // printf("nnz_ndims %d \n", tsr->nnz_ndims[mode]);
        // printf("alldim_nnzs %d \n", tsr->alldim_nnzs[mode].nlens);
        for(tnsIndex i = 0; i < tsr->nnz_ndims[mode]; ++i){
            // determine if this dimension size is less then sparse threshold 
            if(tsr->alldim_nnzs[mode].values[i] < sp_threshold){
                ++spa_dim_num;
            }
        }
        tsr->ndim_blocks[mode] = ndim_block_sizes[mode];
        // printf("spa_dim_num %d success!\n", spa_dim_num);
        // 当全属于稀疏维度时
        if(tsr->nnz_ndims[mode] == spa_dim_num){
            tsr->ndim_count_blocks.values[mode] = 1;
        // 当全属于稠密维度时
        }else if(spa_dim_num == 0){
            tsr->ndim_count_blocks.values[mode] = (tsr->nnz_ndims[mode] - 1) / ndim_block_sizes[mode] + 1;

        }else{
            tsr->ndim_count_blocks.values[mode] = ((tsr->nnz_ndims[mode] - spa_dim_num - 1) / ndim_block_sizes[mode] + 1) + 1;

        }
        tsr->tiles_num *= tsr->ndim_count_blocks.values[mode];
    }
    
    // printf("tiles_num %d \n", tsr->tiles_num);
    // 根据tiles_num开辟相关空间
    tnsFreeIndexVector(&tsr->tile_ptr_begin);
    tnsNewIndexVector(&tsr->tile_ptr_begin, tsr->tiles_num);
    tnsFreeIndexVector(&tsr->tile_ptr_end);
    tnsNewIndexVector(&tsr->tile_ptr_end, tsr->tiles_num);
    

    // 记录每个tile包含的非零元索引
    tnsIndexVector* tile_nnz_index = malloc(tsr->tiles_num * sizeof *tile_nnz_index);
    for(tnsIndex ti = 0; ti < tsr->tiles_num; ++ti){
        // 设置空间大小，初始化为0
        tnsNewIndexVector(&tile_nnz_index[ti], 0);
    }
    // 预处理dim_step
    // tnsIndexVector tsr->ndim_step_blocks;
    // tnsNewIndexVector(&tsr->ndim_step_blocks, tsr->nmodes);
    tsr->ndim_step_blocks.values[tsr->nmodes - 1] = 1;
    for(int mode = tsr->nmodes - 1; mode > 0; --mode){
        tsr->ndim_step_blocks.values[mode-1] = tsr->ndim_step_blocks.values[mode] * tsr->ndim_count_blocks.values[mode]; 
    }
    // 初始化layer的序列
    tsr->layer_dim = 0;
    // 按tsr->layer_dim对应的layer数量分配内存
    free(tsr->layer_tiles);
    tsr->layer_tiles = malloc(tsr->ndim_count_blocks.values[tsr->layer_dim] * sizeof *tsr->layer_tiles);
    for(tnsIndex tile_i = 0; tile_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++tile_i){
        tnsNewIndexVector(&tsr->layer_tiles[tile_i], 0);
    }
    // tnsFreeIndexVector(&tsr->layer_ptr);

    // tnsNewIndexVector(&tsr->layer_ptr, tsr->ndim_count_blocks.values[tsr->layer_dim]+1);
    // tnsContinuousIndexVector(&tsr->layer_ptr, 0, tsr->ndim_step_blocks.values[0]);

    tnsIndex tile_dim_no = 0;
    for(tnsIndex i = 0; i < tsr->nnz; ++i){
        // tnsIndexVector tile_coo;
        // tnsNewIndexVector(&tile_coo, tsr->nmodes);
        tnsIndex tile_full_coo = 0;
        // 展开 coo 格式为一维以判断在哪个tile
        for(int mode = tsr->nmodes - 1; mode >= 0; --mode){
            tile_dim_no = tsr->inds[mode].values[i] / tsr->ndim_blocks[mode];
            // 当属于稀疏块时,超出最大块限制仍记为最后一块
            if(tile_dim_no >= tsr->ndim_count_blocks.values[mode]){
                tile_dim_no = tsr->ndim_count_blocks.values[mode] - 1;
            }
            tile_full_coo += tile_dim_no * tsr->ndim_step_blocks.values[mode];
        }
        // 将当前位置添加到对应的 tile 向量中
        tnsAppendIndexVector(&tile_nnz_index[tile_full_coo], i);
    }
    // 按块顺序遍历tile重序
    // 直接copy一个张量保存原来的数据
    tnsTileSpatsr bak_tsr;
    tnsNewTileSpatsr(&bak_tsr, tsr->ndims, tsr->nmodes, tsr->nnz);
    tnsEasyCopyTileSpatsr(&bak_tsr, tsr);
    tnsIndex tsr_nnz_cur = 0;
    tnsIndex layer_cur = 0;


    // 遍历所有tile, 获得正确的非零元顺序。以及获得每个layer中非零tile的索引
    for(tnsIndex ti = 0; ti < tsr->tiles_num; ++ti){
        tsr->tile_ptr_begin.values[ti] = tsr_nnz_cur;
        // 遍历该tile中每个nnz，根据 tile 的顺序，调整 nnz 的顺序。
        for(tnsIndex nnz_ti = 0; nnz_ti < tile_nnz_index[ti].nlens; ++nnz_ti){
            tsr->values.values[tsr_nnz_cur] = bak_tsr.values.values[tile_nnz_index[ti].values[nnz_ti]];
            for(int mode = 0; mode < tsr->nmodes; ++mode){
                tsr->inds[mode].values[tsr_nnz_cur] = bak_tsr.inds[mode].values[tile_nnz_index[ti].values[nnz_ti]];
            }
            ++tsr_nnz_cur;
        }
        tsr->tile_ptr_end.values[ti] = tsr_nnz_cur;
        // 将不是全0的tile保存在对应的layer_tiles中
        if(tsr->tile_ptr_begin.values[ti] < tsr->tile_ptr_end.values[ti]){
            tnsAppendIndexVector(&tsr->layer_tiles[layer_cur], ti);
        }
        // 每隔 n 个tile 才执行 layer+1
        if((ti+1) % tsr->ndim_step_blocks.values[0] == 0){
            ++layer_cur;
        }
        
    }
    tnsFreeTileSpatsr(&bak_tsr);
    if(tsr->nnz != tsr_nnz_cur){
        printf("tnsTilingTileSpatsr tile 重序出错!\n");
        printf("nnz=%d nnz_chk=%d\n", tsr->nnz, tsr_nnz_cur);    
    }
    if(tsr->ndim_count_blocks.values[tsr->layer_dim] != layer_cur){
        printf("tnsTilingTileSpatsr tile 重序出错!\n");
        printf("layer_num=%d layer_cur=%d\n", tsr->ndim_count_blocks.values[tsr->layer_dim], layer_cur);    
    }

	return 0;
}

/// 根据计算的dim放在最外层的原则，对tile的顺序进行排序，按照初始ndims的顺序执行。
/// ndim_step_blocks,layer_dim,layer_tiles,tile_ptr_begin,tile_ptr_end会改变,其他不变。
int tnsRetilingTileSpatsr(tnsTileSpatsr *tsr, tnsIndex const cur_mode){
    tsr->layer_dim = cur_mode;
    // layer_dim 改变， layer_tiles立即失效
    free(tsr->layer_tiles);
    tsr->layer_tiles = malloc(tsr->ndim_count_blocks.values[tsr->layer_dim] * sizeof *tsr->layer_tiles);
    for(tnsIndex tile_i = 0; tile_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++tile_i){
        tnsNewIndexVector(&tsr->layer_tiles[tile_i], 0);
    }
    
	// 用于保存维度的块数量的交换（方便遍历，不更改源数据）
    tnsIndexVector temp_ndim_count_blocks;
    tnsNewIndexVector(&temp_ndim_count_blocks, tsr->nmodes);
    tnsCopyIndexVector(&temp_ndim_count_blocks, &tsr->ndim_count_blocks);
    temp_ndim_count_blocks.values[0] = tsr->ndim_count_blocks.values[tsr->layer_dim];
    temp_ndim_count_blocks.values[tsr->layer_dim] = tsr->ndim_count_blocks.values[0];

    // 预处理dim_step,保存原step，并计算得到新顺序的step
    tnsIndexVector temp_ndim_step_blocks;
    tnsNewIndexVector(&temp_ndim_step_blocks, tsr->nmodes);
    tnsCopyIndexVector(&temp_ndim_step_blocks, &tsr->ndim_step_blocks);
    tsr->ndim_step_blocks.values[tsr->nmodes - 1] = 1;
    for(int mode = tsr->nmodes - 1; mode > 0; --mode){
        tsr->ndim_step_blocks.values[mode-1] = tsr->ndim_step_blocks.values[mode] * temp_ndim_count_blocks.values[mode]; 
    }


    // 根据维度位置进行块顺序的转换
    // 备份tile的值
    tnsIndexVector temp_tile_ptr_begin;
    tnsNewIndexVector(&temp_tile_ptr_begin, tsr->tiles_num);
    tnsCopyIndexVector(&temp_tile_ptr_begin, &tsr->tile_ptr_begin);
    tnsIndexVector temp_tile_ptr_end;
    tnsNewIndexVector(&temp_tile_ptr_end, tsr->tiles_num);
    tnsCopyIndexVector(&temp_tile_ptr_end, &tsr->tile_ptr_end);

    tnsIndexVector tile_coo;
    tnsNewIndexVector(&tile_coo, tsr->nmodes);
    tnsIndex remainder;
    tnsIndex temp;
    // tnsIndex layer_cur = 0;
    for(tnsIndex i = 0; i < tsr->tiles_num; ++i){
        remainder = i;
        // 还原coo坐标
        for(int mode = 0; mode < tsr->nmodes; ++mode){
            tile_coo.values[mode] = remainder / temp_ndim_step_blocks.values[mode];
            remainder = remainder % temp_ndim_step_blocks.values[mode];
        }
        // 交换coo维度
        temp = tile_coo.values[cur_mode];
        tile_coo.values[cur_mode] = tile_coo.values[0];
        tile_coo.values[0] = temp;
        remainder = 0;
        // coo重新生成一维坐标
        for(int mode = 0; mode < tsr->nmodes; ++mode){
            remainder += tile_coo.values[mode] * tsr->ndim_step_blocks.values[mode];
        }
        // printf(" %d ", remainder);
        tsr->tile_ptr_begin.values[remainder] = temp_tile_ptr_begin.values[i];
        tsr->tile_ptr_end.values[remainder] = temp_tile_ptr_end.values[i];

        // 将不是全0的tile保存在对应的layer_tiles中
        if(tsr->tile_ptr_begin.values[remainder] < tsr->tile_ptr_end.values[remainder]){
            tnsAppendIndexVector(&tsr->layer_tiles[remainder/tsr->ndim_step_blocks.values[0]], remainder);
        }
       
    }

    tnsFreeIndexVector(&temp_tile_ptr_begin);
    tnsFreeIndexVector(&temp_tile_ptr_end);
    
    // tile 重序完成后初始化layer的序列
    // tnsFreeIndexVector(&tsr->layer_ptr);
    // tnsNewIndexVector(&tsr->layer_ptr, tsr->ndim_count_blocks.values[tsr->layer_dim]+1);
    // tnsContinuousIndexVector(&tsr->layer_ptr, 0, tsr->ndim_step_blocks.values[0]);
    tnsFreeIndexVector(&temp_ndim_step_blocks);

    // tnsFreeIndexVector(&modes_vec);
    return 0;
}

