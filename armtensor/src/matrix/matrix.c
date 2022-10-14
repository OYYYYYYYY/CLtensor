
#include <TArm.h>
#include <stdlib.h>
#include <string.h>

/*
稀疏矩阵部分
*/

// todo 暂未发现用途
int tnsNewSparseMatrix(tnsSparseMatrix *mtx, tnsIndex const nrows, tnsIndex const ncols)
{
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    //mtx->stride = ((mtx->ncols+7)>>3)<<3;
    mtx->nnz = 0;
    tnsNewIndexVector(&mtx->rowinds, 0);
    tnsNewIndexVector(&mtx->colinds, 0);
    tnsNewValueVector(&mtx->values, 0);
    return 0;

}

//释放空间
void tnsFreeSparseMatrix(tnsSparseMatrix *mtx)
{
    mtx->nrows = 0;
    mtx->ncols = 0;
    //mtx->stride = 0;
    mtx->nnz = 0;
    tnsFreeIndexVector(&mtx->rowinds);
    tnsFreeIndexVector(&mtx->colinds);
    tnsFreeValueVector(&mtx->values);
}

// 从文件中读取稀疏矩阵
int tnsLoadSparseMatrix(tnsSparseMatrix *mtx, FILE *fp){
    fscanf(fp, "%u", &mtx->nrows);
    fscanf(fp, "%u\n", &mtx->ncols);
    //mtx->stride=((mtx->ncols+7)>>3)<<3;
    tnsIndex nnz = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    long fp_local = ftell(fp);
    //获得非零元个数
    while((getline(&line,&len,fp)) != -1){
        ++nnz;
    }
    mtx->nnz = nnz;
    tnsNewIndexVector(&mtx->rowinds, nnz);
    tnsNewIndexVector(&mtx->colinds, nnz);
    tnsNewValueVector(&mtx->values, nnz);

    tnsIndex nnz_chk = 0;
    char *ptr = NULL;
    fseek(fp, fp_local, 0);
    while((getline(&line,&len,fp)) != -1){
        mtx->rowinds.values[nnz_chk] = strtoul(line, &ptr, 10);
        mtx->colinds.values[nnz_chk] = strtoul(ptr, &ptr, 10);
        mtx->values.values[nnz_chk] = strtod(ptr, &ptr);
        ++nnz_chk;

    }
    if(nnz != nnz_chk) printf("getline函数出错");
    return 0;
}
// 保存稀疏矩阵到文件
int tnsDumpSparseMatrix(tnsSparseMatrix *mtx, FILE *fp){
    fprintf(fp, "%u %u \n", mtx->nrows, mtx->ncols);
    for(tnsIndex i = 0; i < mtx->nnz; ++i)
    {
        fprintf(fp, "%u %u %lf\n", mtx->rowinds.values[i], mtx->colinds.values[i], mtx->values.values[i]);
    }
    return 0;
}











/*
稠密矩阵部分
*/
///新建一个空的矩阵
int tnsNewDenseMatrix(tnsDenseMatrix *mtx, tnsIndex const nrows, tnsIndex const ncols){
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    int blk = 4;
    mtx->stride = ((mtx->ncols + blk - 1) / blk) * blk;
    tnsNewValueVector(&mtx->values, mtx->nrows * mtx->stride);
    return 0;
}

// 从文件读入
int tnsLoadDenseMatrix(tnsDenseMatrix *mtx, FILE *fp)
{
    //读入矩阵的维度
    fscanf(fp,"%u", &mtx->nrows);
    fscanf(fp,"%u", &mtx->ncols);
    // tnsNewValueVector(&mtx->values, mtx->nrows * mtx->stride);
    tnsNewDenseMatrix(mtx, mtx->nrows, mtx->ncols);
    
    for(tnsIndex i = 0; i < mtx->nrows; ++i)
    {
        for(tnsIndex j = 0; j < mtx->ncols; ++j)
            fscanf(fp, "%f", &mtx->values.values[i * mtx->stride + j]);
    }
    return 0;
}

void tnsFreeDenseMatrix(tnsDenseMatrix *mtx)
{
    mtx->nrows = 0;
    mtx->ncols = 0;
    mtx->stride = 0;
    tnsFreeValueVector(&mtx->values);
}

int tnsDumpDenseMatrix(tnsDenseMatrix *mtx, FILE *fp)
{
    fprintf(fp, "%u %u\n", mtx->nrows, mtx->ncols);
    for(tnsIndex i = 0; i < mtx->nrows; ++i)
    {
        for(tnsIndex j = 0; j < mtx->ncols; ++j)
        {
            fprintf(fp, "%lf ", mtx->values.values[i * mtx->stride + j]);
        }
        fprintf(fp, "\n");
    }
   
    return 0;
}

// 赋常数值
int tnsConstantDenseMatrix(tnsDenseMatrix * const mtx, tnsValue const val){
    tnsIndex nnz = mtx->nrows * mtx->stride;
    // for(tnsIndex i = 0; i < mtx->nrows; ++i)
    // {
    //     for(tnsIndex j = 0; j < mtx->ncols; j++)
    //     {
    //         mtx->values.values[i * mtx->stride + j] = val;
    //     }
    // }
    #pragma omp parallel for num_threads(32)
    for(tnsIndex i = 0; i < nnz; ++i){
        mtx->values.values[i] = val;
    }
    return 0;
}
static tnsValue tnsRandomValue(void){
  tnsValue v =  3.0 * ((tnsValue) rand() / (tnsValue) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}

// 赋随机值0-1,随机值时间太长了
int tnsRandomDenseMatrix(tnsDenseMatrix * const mtx){
    // for(tnsIndex i = 0; i < mtx->nrows; ++i){
    //     for(tnsIndex j = 0; j < mtx->ncols; j++){
    //         mtx->values.values[i * mtx->stride + j] = tnsRandomValue();
    //     }
    // }
    tnsIndex nnz = mtx->nrows * mtx->stride;
    #pragma omp parallel for num_threads(32)
    for(tnsIndex i = 0; i < nnz; ++i){
        mtx->values.values[i] = tnsRandomValue();
    }
    
    return 0;
}

// 拷贝Matrxi
int tnsCopyDenseMatrix(tnsDenseMatrix *dest, const tnsDenseMatrix *src)
{
    if(dest->nrows != src->nrows || dest->ncols != src->ncols ){
        printf("维数不匹配\n");
        return 1;
    }
    
    memcpy(dest->values.values, src->values.values, src->nrows * src->stride * sizeof *src->values.values);

    return 0; 

}




