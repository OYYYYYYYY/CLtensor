#include <TArm.h>
#include <stdlib.h>

static int CMP_QuickSort(tnsSparseTensor *A, tnsIndex i, tnsIndex* key_index){
    for(tnsIndex k = 0; k < A->nmodes; ++k){
        //if(A->inds[k].values[i] < key_index[k]) return 0;
        if(A->inds[k].values[i] == key_index[k]) continue;
        return A->inds[k].values[i] > key_index[k];
    }
   
}

static int CMP_QuickSort_1(tnsSparseTensor *A, tnsIndex i, tnsIndex* key_index){
    for(tnsIndex k = 0; k < A->nmodes; ++k){
        if(key_index[k] < A->inds[k].values[i]) return 0;
    }
    return 1;
}

/// multidimensional Valuation, ith element <- jth element
static void Mult_Valuation(tnsSparseTensor *A, tnsIndex i, tnsIndex j){
    tns_CheckError(i > A->nnz || j > A->nnz, "Mult_Valuation", "超出张量非零元的范围");
    for(tnsIndex k = 0; k < A->nmodes; ++k)
        A->inds[k].values[i]=A->inds[k].values[j];
    A->values.values[i] = A->values.values[j];
}





static void tnsQuickSortSpatsr(tnsSparseTensor *A, int begin, int end){
    if(begin < end){
        tnsIndex *key_index = calloc(A->nmodes,sizeof(tnsIndex));
        tnsValue key_val;
        //SaveNnz(A, key_index, begin, &key_val);
        for(tnsIndex i=0;i<A->nmodes;++i) key_index[i]=A->inds[i].values[begin];
        key_val=A->values.values[begin];
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
}



/**
  * @fn tnsPermuteSpatsr
  * @brief 对张量进行转置和排序
  * @details 对A根据order进行转置，然后对nnz进行排序
  * @param[in]  A 原始张量
  * @param[in]  order 对A进行转置的order的顺序
  * @param[out] B 根据order进行处理的A，使用新的空间去存储参数
  * @return 返回转置和排序后的张量
  */
int tnsPermuteSpatsr(tnsSparseTensor *B, tnsSparseTensor *A, tnsIndex *order){
    tnsCopySparseTensor(B, A);
    tnsIndexVector *temp_inds = malloc(B->nmodes * sizeof *temp_inds);
    memcpy(temp_inds, B->inds, B->nmodes * sizeof *temp_inds);
    //permute
    for(tnsIndex i = 0; i < A->nmodes; ++i){
        B->ndims[i] = A->ndims[order[i]];
        B->inds[i] = temp_inds[order[i]];
    }
    //sort
    tnsQuickSortSpatsr(B, 0, B->nnz-1);
    
    free(temp_inds);
    return 0;
}



