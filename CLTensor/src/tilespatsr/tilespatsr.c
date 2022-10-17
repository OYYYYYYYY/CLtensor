
#include <TArm.h>
#include <stdlib.h>
#include <string.h>


/*
Sparse Tile Tensor Base Function
*/
// Initialize a TileSpatsr
int tnsNewTileSpatsr(tnsTileSpatsr *tsr, const tnsIndex *ndims, tnsIndex nmodes, tnsIndex nnz){
    tsr->nmodes = nmodes;
    tsr->ndims = calloc(nmodes, sizeof *tsr->ndims);
    tsr->nnz_ndims = calloc(nmodes, sizeof *tsr->nnz_ndims);
    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    tsr->alldim_raw_indexs = malloc(tsr->nmodes * sizeof *tsr->alldim_raw_indexs);
    tsr->alldim_nnzs = malloc(tsr->nmodes * sizeof *tsr->alldim_nnzs);
    tsr->nnz = nnz;
    // tsr->alldim_num = 0;
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    memcpy(tsr->nnz_ndims, ndims, nmodes * sizeof *tsr->nnz_ndims);
    for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){
        tnsNewIndexVector(&tsr->inds[mode], tsr->nnz);
        tnsNewIndexVector(&tsr->alldim_raw_indexs[mode], tsr->ndims[mode]);
        tnsNewIndexVector(&tsr->alldim_nnzs[mode], tsr->ndims[mode]);
        // tsr->alldim_num += tsr->ndims[mode];
    }
    tnsNewValueVector(&tsr->values, tsr->nnz);

    // tsr->alldim_raw_indexs = calloc(tsr->alldim_num, sizeof *tsr->alldim_raw_indexs);
    // tsr->alldim_nnzs = calloc(tsr->alldim_num, sizeof *tsr->alldim_nnzs);
    tsr->ndim_blocks = calloc(nmodes, sizeof *tsr->ndim_blocks);
    tnsNewIndexVector(&tsr->ndim_count_blocks, nmodes);
    tnsNewIndexVector(&tsr->ndim_step_blocks, nmodes);
    tsr->tiles_num = 0;
    tnsNewIndexVector(&tsr->tile_ptr_begin, tsr->tiles_num);
    tnsNewIndexVector(&tsr->tile_ptr_end, tsr->tiles_num);
    tsr->layer_dim = 0;
    tsr->layer_tiles = malloc(tsr->ndim_count_blocks.values[tsr->layer_dim] * sizeof *tsr->layer_tiles);
    // for(tnsIndex tile_i = 0; tile_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++tile_i){
    //     tnsNewIndexVector(&tsr->layer_tiles[tile_i], 0);
    //     // tsr->alldim_num += tsr->ndims[mode];
    // }
    // tnsNewIndexVector(&tsr->layer_ptr, tsr->ndim_count_blocks.values[tsr->layer_dim]+1);

    return 0;
}


// from SparseTensor to TileSpatsr
int tnsSparseTensor2TileSpatsr(tnsTileSpatsr *dest, const tnsSparseTensor *src){
    tns_CheckError(dest->nmodes != src->nmodes, "tnsSparseTensor2TileSpatsr", "tensor order dismatch");
    tns_CheckError(dest->nnz != src->nnz, "tnsSparseTensor2TileSpatsr", "tensor nnz dismatch");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    for(tnsIndex i=0; i < src->nmodes; ++i){
        tnsCopyIndexVector(&dest->inds[i], &src->inds[i]);
    }
    tnsCopyValueVector(&dest->values, &src->values);
    return 0;
}


// Free a TileSpatsr
void tnsFreeTileSpatsr(tnsTileSpatsr *tsr){
    // 因为New的时候没有创建空间，所以释放前需要先判断
    if(tsr->ndim_count_blocks.values[tsr->layer_dim] != 0){
        for(tnsIndex tile_i = 0; tile_i < tsr->ndim_count_blocks.values[tsr->layer_dim]; ++tile_i){
            tnsFreeIndexVector(&tsr->layer_tiles[tile_i]);
            // tsr->alldim_num += tsr->ndims[mode];
        }
    }
    
    free(tsr->layer_tiles);
    // tnsFreeIndexVector(&tsr->layer_ptr);
    tsr->layer_dim = 0;
    tnsFreeIndexVector(&tsr->tile_ptr_end);
    tnsFreeIndexVector(&tsr->tile_ptr_begin);
    tsr->tiles_num = 0;

    // free(tsr->alldim_raw_indexs);
    // free(tsr->alldim_nnzs);
    free(tsr->ndim_blocks);
    tnsFreeIndexVector(&tsr->ndim_count_blocks);
    tnsFreeIndexVector(&tsr->ndim_step_blocks);
    
    free(tsr->ndims);
    free(tsr->nnz_ndims);
    tnsFreeValueVector(&tsr->values);
    
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        tnsFreeIndexVector(&tsr->inds[i]);
        tnsFreeIndexVector(&tsr->alldim_raw_indexs[i]);
        tnsFreeIndexVector(&tsr->alldim_nnzs[i]);
    }
    free(tsr->inds);
    free(tsr->alldim_raw_indexs);
    free(tsr->alldim_nnzs);
    tsr->nmodes = 0;
    // tsr->alldim_num = 0;
    tsr->nnz = 0;
}


int tnsDumpTileSpatsr(tnsTileSpatsr *tsr, FILE *fp){
    fprintf(fp, "%u\n", tsr->nmodes);
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        fprintf(fp, "%u ", tsr->ndims[i]);
    }
    fprintf(fp, "\n");
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        fprintf(fp, "%u ", tsr->ndim_count_blocks.values[i]);
    }
    fprintf(fp, "\n");
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        fprintf(fp, "%u ", tsr->ndim_step_blocks.values[i]);
    }
    fprintf(fp, "\n");
    // for(tnsIndex i = 0; i < tsr->nmodes; ++i){
    //   for(tnsIndex j = 0; j < tsr->ndims[i]; ++j){
    //     fprintf(fp, "%u\n", tsr->alldim_nnzs[i].values[j]);
    //   }
    // }
    fprintf(fp, "\n");
    // for(tnsIndex i = 0; i < tsr->tiles_num; ++i){
    //     fprintf(fp, "%u ", tsr->tile_ptr_begin.values[i]);
    // }
    // fprintf(fp, "\n");
    // for(tnsIndex i = 0; i < tsr->tiles_num; ++i){
    //     fprintf(fp, "%u ", tsr->tile_ptr_end.values[i]);
    // }
    // fprintf(fp, "\n");
    // for(tnsIndex i = 0; i < tsr->ndim_count_blocks.values[tsr->layer_dim]+1; ++i){
    //     fprintf(fp, "%u ", tsr->layer_ptr.values[i]);
    // }
    fprintf(fp, "\n%u %u\n", tsr->tiles_num, tsr->nnz);
    // for(tnsIndex i = 0; i < tsr->nnz; ++i){
    //     for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){
    //         fprintf(fp, "%u ", tsr->inds[mode].values[i]);
    //     }
    //     fprintf(fp, "%f \n", tsr->values.values[i]);
    // }
    
    return 0;
}


int tnsLoadTileSpatsr(tnsTileSpatsr *tsr, FILE *fp){
    char *line = NULL;
    size_t len = 0;
    
    //读入张量的阶数
    tnsIndex nmodes_temp;
    tnsIndex *ndims_temp;
    fscanf(fp, "%u", &nmodes_temp);
    ndims_temp = malloc(nmodes_temp * sizeof(tnsIndex));
    tnsIndex mode;
    tnsIndex nnz = 0;
    for(mode = 0; mode < nmodes_temp; ++mode){
        fscanf(fp, "%u", &ndims_temp[mode]);
    }
    fscanf(fp,"\n");
    long fp_local = ftell(fp);
    //获得非零元个数
    while((getline(&line, &len, fp)) != -1){
        ++nnz;
    }
    
    tnsNewTileSpatsr(tsr, ndims_temp, nmodes_temp, nnz);
    free(ndims_temp);

    //为nnz开辟val和coo index空间
    for(mode=0; mode < tsr->nmodes; ++mode){
        tnsNewIndexVector(&tsr->inds[mode], tsr->nnz);
    }
    tnsNewValueVector(&tsr->values, tsr->nnz);

    tnsIndex nnz_chk = 0;
    char *ptr = NULL;
    fseek(fp, fp_local, 0);
    while((getline(&line, &len, fp)) != -1){
        ptr = line;
        // tnsIndex alldim_nnzs_index_ptr = 0;
        for(mode = 0; mode < tsr->nmodes; ++mode){
            tnsIndex dim_index = strtoul(ptr, &ptr, 10);
            tsr->inds[mode].values[nnz_chk] = dim_index;
            // 统计每个dim的非零元的数目 
            tsr->alldim_nnzs[mode].values[dim_index] += 1;
            // tsr->alldim_nnzs[alldim_nnzs_index_ptr + dim_index] += 1;
            // alldim_nnzs_index_ptr += tsr->ndims[mode];
        }
        tsr->values.values[nnz_chk] = strtod(ptr, &ptr);
        ++nnz_chk;
    }



    if(nnz != nnz_chk){
        printf("getline函数出错\n");
        printf("nnz=%d nnz_chk=%d\n", nnz, nnz_chk);    
    }

    return 0;
}


int tnsEasyCopyTileSpatsr(tnsTileSpatsr *dest, const tnsTileSpatsr *src){
    tns_CheckError(dest->nmodes != src->nmodes, "tnsCopySparseTensor", "tensor order dismatch");
    tns_CheckError(dest->nnz != src->nnz, "tnsCopySparseTensor", "tensor nnz dismatch");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    for(tnsIndex i=0; i < src->nmodes; ++i){
        tnsCopyIndexVector(&dest->inds[i], &src->inds[i]);
    }
    tnsCopyValueVector(&dest->values, &src->values);
    return 0;
}

