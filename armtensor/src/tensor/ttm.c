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

// 稀疏张量乘以稠密矩阵（行乘），得到稠密张量c,c 需要全零初始化
// int tnsTTMSpatsr(tnsDenseTensor *dentsr_c, const tnsSparseTensor *spatsr_a, tnsDenseMatrix *denmat_b, tnsIndex copt_mode){
//     // 输入参数检查
//     tns_CheckError((spatsr_a->ndims[copt_mode] != denmat_b->nrows), "tnsTTMSpatsr", "相乘维度不匹配");

//     // 张量index_mode相同， copt_mode 变化。张量c除了 copt_mode 外，其他模态都与 a 相同
//     // 执行乘法计算 : 外积累加的计算方式，每个张量 a 的非零元乘以矩阵 b 对应的一列， 得到可以累加的列
//     tnsIndex hash_index = 0; ///< 记录当前非零元 属于 稠密结果对应的位置
//     tnsIndex hash_step = 1; ///< 记录当前非零元 属于 稠密结果对应的位置
//     tnsValue ele_a;
//     tnsValue ele_b;
//     // 遍历所有张量非零元
//     for(tnsIndex nnz_i = 0; nnz_i < spatsr_a->nnz; ++nnz_i){
//         ele_a = spatsr_a->values.values[nnz_i];

//         // 每个张量非零元都需要与矩阵的某一列所有元素计算，所以遍历行
//         for(tnsIndex row_i = 0; row_i < denmat_b->nrows; ++row_i){
//             // 稠密矩阵的对应方式
//             ele_b = denmat_b->values.values[ row_i * denmat_b->stride + spatsr_a->inds[copt_mode].values[nnz_i]];
//             //printf("%0.f ", ele_b);
//             hash_index = 0;
//             hash_step = 1;

//             // 每个非零元计算 mode hash结果（可优化）从后面模态遍历到前面模态。
//             for(int mode_i = spatsr_a->nmodes-1; mode_i >= 0; --mode_i){
//                 if(mode_i != (int)copt_mode){ ///< index_mode 按张量 a 的
//                     hash_index += spatsr_a->inds[mode_i].values[nnz_i] * hash_step;
//                     hash_step *= spatsr_a->ndims[mode_i];
//                 }else{ ///< copt_mode 按矩阵 b 的
//                     hash_index += row_i * hash_step;
//                     hash_step *= denmat_b->nrows;
//                 }
//             }
//             // 将计算结果保存到对应位置
//             dentsr_c->values.values[hash_index] += ele_a * ele_b; 
//         } ///< for row_i
//     } ///< for nnz_i

//     return 0;

// }