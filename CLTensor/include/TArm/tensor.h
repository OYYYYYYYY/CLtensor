#ifndef TARM_TENSOR_H
#define TARM_TENSOR_H


/*
稠密张量部分
*/
// 分配空间
int tnsNewDenseTensor(tnsDenseTensor *tsr, tnsIndex nmodes, const tnsIndex ndims[]);
// 释放空间
void tnsFreeDenseTensor(tnsDenseTensor *tsr);
// 构建常量张量
int tnsConstantDenseTensor(tnsDenseTensor * const tsr, tnsValue const val);
// 从文件读入稠密张量
int tnsLoadDenseTensor(tnsDenseTensor *tsr,FILE *fp);
// 将稠密张量写入文件
int tnsDumpDenseTensor(tnsDenseTensor *tsr,FILE *fp);


/*
稀疏张量部分
*/
// 分配空间
int tnsNewSparseTensor(tnsSparseTensor *tsr, const tnsIndex *ndims, tnsIndex nmodes, tnsIndex nnz);
void tnsFreeSparseTensor(tnsSparseTensor *tsr);

// 赋值&储存模块
// 赋常数值

// 拷贝Tensor
int tnsCopySparseTensor(tnsSparseTensor *dest, const tnsSparseTensor *src);
// 从文件读入稀疏张量
int tnsLoadSparseTensor(tnsSparseTensor *tsr, FILE *fp);
// 存入文件稀疏张量
int tnsDumpSparseTensor(tnsSparseTensor *tsr, FILE *fp);

// Get属性模块（长度，范式，秩等等）



// 张量操作模块（包括矩阵乘加，矩阵转置等等）



#endif