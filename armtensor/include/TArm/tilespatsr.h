
#ifndef TARM_TILESPATSR_H
#define TARM_TILESPATSR_H


/**
 * Tile Sparse Tensor Format
 */

typedef struct {
    tnsIndex nmodes;      /// # modes
    tnsIndex *ndims;      /// size of each mode, length nmodes
    tnsIndex nnz;         /// 
    tnsIndexVector *inds;       /// indices of each element, length [nmodes][nnz]
    tnsValueVector values;      /// non-zero values, length nnz
    /// sparse sort 相关的属性
    tnsIndex *nnz_ndims;      /// size of each mode, length nmodes, deprecated none
    tnsIndexVector *alldim_raw_indexs; ///  record the mapping of raw ndims.[nmodes][ndims[mode]]，以前的第i个数，对应排序后的位置val
    tnsIndexVector *alldim_nnzs; ///  nnz of per dim. create in load [nmodes][ndims[mode]]
    /// tiling 相关的属性
    tnsIndex *ndim_blocks; ///  block size of nmodes.[nmodes]
    tnsIndexVector ndim_count_blocks; ///  number of blocks per nmodes.[nmodes]
    tnsIndex tiles_num;    /// tile number of tensor.
    /// 会随着 retiling 而改变的属性
    tnsIndexVector ndim_step_blocks;   /// step of blocks per nmodes.[nmodes]
    tnsIndexVector tile_ptr_begin;   /// list the nnz start offset of tiles.[tiles_num]
    tnsIndexVector tile_ptr_end;   /// list the nnz end offset of tiles.[tiles_num]
    tnsIndex layer_dim;    /// layer number of tensor.
    tnsIndexVector *layer_tiles;   /// list the tile offset of layers.[ndim_count_blocks[layer_dim]][tile_num]
} tnsTileSpatsr;

int tnsNewTileSpatsr(tnsTileSpatsr *tsr, const tnsIndex *ndims, tnsIndex nmodes, tnsIndex nnz);

void tnsFreeTileSpatsr(tnsTileSpatsr *tsr);

int tnsLoadTileSpatsr(tnsTileSpatsr *tsr, FILE *fp);

int tnsDumpTileSpatsr(tnsTileSpatsr *tsr, FILE *fp);

int tnsEasyCopyTileSpatsr(tnsTileSpatsr *dest, const tnsTileSpatsr *src);

/// 根据 ndims 的顺序，从大到小的重新对nnz排序。 目的是：密集dim的在前面，稀疏dim的在后面。
int tnsPermuteTileSpatsr(tnsTileSpatsr *tsr, tnsIndex *order);

/// dim重序，再根据新的dim对nnz重序。
/// 先生成dim的对应，再修改nnz中对应的数值。根据新的dim顺序更改nnz的排列顺序。
int tnsRedimTileSpatsr(tnsTileSpatsr *tsr);
// int tnsReorderTileSpatsr(tnsTileSpatsr *tsr, tnsIndex *ndim_block_sizes);

/// 根据cache的大小，划分成tile，并根据tile顺序对nnz排序，ndims的顺序是tile之间的顺序。
int tnsTilingTileSpatsr(tnsTileSpatsr *tsr, tnsIndex *ndim_block_sizes, tnsIndex sp_threshold);

/// 根据计算的dim放在最外层的原则，对tile的顺序进行排序，按照初始ndims的顺序执行。
int tnsRetilingTileSpatsr(tnsTileSpatsr *tsr, tnsIndex const mode);


/// 分tile计算MTTKRP（弃用）
int tnsMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix *U_list);

/// MTTKRP OMP版本
int tnsOmpMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const mode, const tnsIndex tk);

/// COO直接计算MTTKRP
int tnsNaiveMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const mode, const tnsIndex tk);
int tnsOMPNaiveMTTKRPTileSpatsr(tnsTileSpatsr *tsr, tnsDenseMatrix **U_list, tnsIndex const mode, const tnsIndex tk);

int tnsVecTilingTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr);

int tnsTTVTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr, tnsValueVector *vec, tnsIndex const copt_mode, const tnsIndex tk);
int tnsOMPTTVTileSpatsr(tnsSparseTensor *Y_tsr, tnsTileSpatsr *X_tsr, tnsValueVector *vec, tnsIndex const copt_mode, const tnsIndex tk);

#endif