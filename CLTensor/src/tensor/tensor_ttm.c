#include <TArm.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#define nthreads 4

int tnsTTMSpatsr(tnsSparseTensor *spatsrY, tnsSparseTensor *spatsrX, tnsDenseMatrix *denmat, tnsIndex copt_mode){
    // Y的模态信息在使用该函数前在外部确定
    if(copt_mode >= spatsrX->nmodes){
        printf("指定维度不存在\n");
        return 1;
    }
    if(spatsrX->ndims[copt_mode] != denmat->nrows || spatsrY->ndims[copt_mode] != denmat->ncols){
        printf("维度不匹配\n");
        return 1;
    }    

    tnsIndex row = 1;
	tnsIndex col = 1;
	for(tnsIndex i = 0; i <= copt_mode; ++i){
        row *= spatsrX->ndims[i];   
	}
    for(tnsIndex i = copt_mode + 1; i < spatsrX->nmodes; ++i){
        col *= spatsrX->ndims[i];   
	}
	tnsSparseMatrix mtx;
	tnsNewSparseMatrix(&mtx, row, col);

	mtx.nnz = spatsrX->nnz;
    mtx.rowinds.nlens = mtx.nnz;
    mtx.colinds.nlens = mtx.nnz;
    mtx.values.nlens = mtx.nnz;
    mtx.rowinds.values = realloc(mtx.rowinds.values, (mtx.nnz) * sizeof mtx.rowinds);
    mtx.colinds.values = realloc(mtx.colinds.values, (mtx.nnz) * sizeof mtx.colinds);
    mtx.values.values = realloc(mtx.values.values, (mtx.nnz) * sizeof mtx.values);
    
    for(tnsIndex nnz_i = 0; nnz_i < spatsrX->nnz; ++nnz_i){
        tnsIndex hash_rowstep = 1;
    	tnsIndex hash_colstep = 1;
    	tnsIndex hash_rowinds = 0;
    	tnsIndex hash_colinds = 0;
		if(nnz_i){
        	for(int mode_i = copt_mode; mode_i >= 0; --mode_i){
        	    hash_rowinds += spatsrX->inds[mode_i].values[nnz_i] * hash_rowstep;
        	    hash_rowstep *= spatsrX->ndims[mode_i]; 
        	}///< for mode_i
        	for(int mode_j = spatsrX->nmodes - 1; mode_j > copt_mode; --mode_j){
        	    hash_colinds += spatsrX->inds[mode_j].values[nnz_i] * hash_colstep;
        	    hash_colstep *= spatsrX->ndims[mode_j];
        	}///< for mode_j
		}
        mtx.rowinds.values[nnz_i] = hash_rowinds;
        mtx.colinds.values[nnz_i] = hash_colinds;
        mtx.values.values[nnz_i] = spatsrX->values.values[nnz_i];
    }///< for nnz_i
    tnsFreeSparseMatrix(&mtx);

    tnsIndex hash_index = 0;
    tnsIndex hash_step = 1;
    tnsIndex indexs = 1;
    tnsValue data_a;
    tnsValue data_b;
    tnsIndex count = 0;
    tnsIndex length = 1;
    tnsIndex II = 1;
    // zhi
	if(spatsrX->nmodes == 3)
		II = spatsrX->nmodes * 8;
    if(spatsrX->nmodes == 4)
        II = spatsrX->nmodes * 7;
	if(spatsrX->nmodes == 5)
		II = spatsrX->nmodes * 6;

    // 获取spatsrY的松散上界
    for(tnsIndex i = 0; i < spatsrY->nmodes; ++i){
        length  = length * spatsrY->ndims[i];
    } 
    // 获取步长strides
    tnsIndex strides = length / spatsrY->ndims[copt_mode];
    
    // 创建Valuevacter datavec存储中间值
    tnsValueVector datavec;
    tnsNewValueVector(&datavec, length);
    
    tnsIndex countX; 
    for(tnsIndex i = 0; i < spatsrX->nnz; ++i)
        if(spatsrX->values.values[i] != 0)
            ++countX;
     
    // 计算中间结果values
	for(tnsIndex i = 0; i < II; ++i){
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
            	datavec.values[indexs] = data_a * data_b;
        	}///< for col_i
    	}///< for nnz_i
	}


    for(tnsIndex i = 0; i < datavec.nlens; ++i)
        if(datavec.values[i] != 0)
            ++count;
    spatsrY->nnz = count;
    
        //通过准确的非零元数重新分配spatsrY的values和inds空间
    tnsFreeValueVector(&spatsrY->values);
    tnsNewValueVector(&spatsrY->values, spatsrY->nnz);
    for(tnsIndex mode = 0; mode < spatsrY->nmodes; ++mode){
        tnsFreeIndexVector(&spatsrY->inds[mode]);
        tnsNewIndexVector(&spatsrY->inds[mode], spatsrY->nnz);
    }

    for(tnsIndex i = 0; i < II; ++i){
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
    }

    tnsFreeValueVector(&datavec);
	
    return 0;
}

int tnsOmpTTMSpatsr(tnsSparseTensor *spatsrY, const tnsSparseTensor *spatsrX, const tnsDenseMatrix *denmat, tnsIndex copt_mode){
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