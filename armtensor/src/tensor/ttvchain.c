#include <TArm.h>
#include <stdlib.h>
#include <string.h>


// typedef struct {
//     tnsIndex nmodes;      /// # modes
//     tnsIndex *ndims;      /// size of each mode, length nmodes
//     tnsIndex nnz;         /// 
//     tnsIndexVector *inds;       /// indices of each element, length [nmodes][nnz]
//     tnsValueVector values;      /// non-zero values, length nnz
// } tnsSparseTensor;
// typedef struct {
//     tnsIndex nrows;      /// # modes
//     tnsIndex ncols;      /// size of each mode, length nmodes
//     tnsIndex stride;    ///是大于ncols的最小8的倍数
//     tnsValueVector values;      /// non-zero values, length rows*cols
// } tnsDenseMatrix;
// typedef struct {
//     tnsIndex nmodes;      /// # modes
//     tnsIndex nnz;         /// nonzero size (multiply by all ndims)
//     tnsIndex *ndims;      /// size of each mode, length nmodes
//     tnsValueVector values;      /// non-zero values, length sum(ndims)
// } tnsDenseTensor;


// int reindexSpatsr(tnsSparseTensor *spatsr_a, tnsIndex copt_mode){

// }

// 稀疏张量乘以多个向量（行乘），得到向量c
// int tnsTTVChainSpatsr(tnsValueVector *vec_c, const tnsSparseTensor *spatsr_a, tnsValueVector *vec_b_list, tnsIndex fix_mode){
//     // 输入参数检查
//     if(spatsr_a->ndims[fix_mode] != vec_c->nlens){
//         tns_CheckError(0, "tnsTTMSpaten", "输出维度不匹配");
//         return 1;
//     }
    



//     return 0;

// }