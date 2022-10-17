
#include <TArm.h>
#include <stdlib.h>
#include <string.h>


/*
稠密张量部分
*/
// 初始化张量,稠密张量开辟ndims和values空间
int tnsNewDenseTensor(tnsDenseTensor *tsr, tnsIndex nmodes, const tnsIndex ndims[])
{
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof(tnsIndex));
    
    memcpy(tsr->ndims, ndims, nmodes * sizeof(tnsIndex));
    tnsIndex nnz = 1;
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        nnz *= tsr->ndims[i];
    }
    tsr->nnz = nnz;
    tnsNewValueVector(&tsr->values, tsr->nnz);
    
    return 0;
}

//释放空间
void tnsFreeDenseTensor(tnsDenseTensor *tsr)
{
    free(tsr->ndims);
    tnsFreeValueVector(&tsr->values);
    
}

//构建常量张量
int tnsConstantDenseTensor(tnsDenseTensor * const tsr, tnsValue const val)
{
    for(tnsIndex i = 0; i < tsr->nnz; ++i){
        tsr->values.values[i] = val;
    }
    return 0;
}

int tnsLoadDenseTensor(tnsDenseTensor *tsr, FILE *fp)
{
    char *line = NULL;
    size_t len = 0;
    tnsIndex nmodes_temp;
    tnsIndex *ndims_temp;
    fscanf(fp, "%u", &nmodes_temp);
    ndims_temp = malloc(nmodes_temp * sizeof(tnsIndex));
    tnsIndex mode;
    
    for(mode = 0; mode < nmodes_temp; ++mode){
        fscanf(fp, "%u", &ndims_temp[mode]);
        
    }
    
    tnsNewDenseTensor(tsr, nmodes_temp, ndims_temp);
    tnsIndex nnz = 0;
   
    while(feof(fp) == 0)
    {
        tnsValue temp;
        fscanf(fp, "%f ", &temp);
        
        tsr->values.values[nnz] = temp;
        ++nnz;
    }
    
    if(nnz != tsr->nnz)
    {
        printf("文件读取出错\n");
        printf("nnz=%d tsr->nnz=%d\n", nnz, tsr->nnz);    
    }
    return 0;
}

int tnsDumpDenseTensor(tnsDenseTensor *tsr, FILE *fp)
{
    fprintf(fp,"%u\n",tsr->nmodes);
    for(tnsIndex i = 0; i < tsr->nmodes; ++i)
    {
        fprintf(fp, "%u ", tsr->ndims[i]);
    }
    fprintf(fp, "\n");
    //fprintf(fp,"\n%u\n",tsr->nnz);
    for(tnsIndex i = 0; i <tsr->nnz; ++i)
    {
        fprintf(fp, "%f ", tsr->values.values[i]);
    }
    fprintf(fp, "\n");
    return 0;
}



/*
稀疏张量
*/
//开辟ndims空间
int tnsNewSparseTensor(tnsSparseTensor *tsr, const tnsIndex *ndims, tnsIndex nmodes, tnsIndex nnz)
 {
    tsr->nmodes = nmodes;
    tsr->ndims = calloc(nmodes, sizeof *tsr->ndims);
    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    tsr->nnz = nnz;
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){
        tnsNewIndexVector(&tsr->inds[mode], tsr->nnz);
    }
    tnsNewValueVector(&tsr->values, tsr->nnz);
    return 0;
}

int tnsDumpSparseTensor(tnsSparseTensor *tsr, FILE *fp)
{
    fprintf(fp, "%u\n", tsr->nmodes);
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        fprintf(fp, "%u ", tsr->ndims[i]);
    }
    fprintf(fp, "\n%u\n", tsr->nnz);
    for(tnsIndex i = 0; i < tsr->nnz; ++i){
        for(tnsIndex mode = 0; mode < tsr->nmodes; ++mode){
            fprintf(fp, "%u ", tsr->inds[mode].values[i]);
        }
        fprintf(fp, "%f \n", tsr->values.values[i]);
    }
    
    return 0;
}


int tnsCopySparseTensor(tnsSparseTensor *dest, const tnsSparseTensor *src)
{
    tns_CheckError(dest->nmodes != src->nmodes, "tnsCopySparseTensor", "tensor order dismatch");
    tns_CheckError(dest->nnz != src->nnz, "tnsCopySparseTensor", "tensor nnz dismatch");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    for(tnsIndex i=0; i < src->nmodes; ++i){
        tnsCopyIndexVector(&dest->inds[i], &src->inds[i]);
    }
    tnsCopyValueVector(&dest->values, &src->values);
    return 0;
}


int tnsLoadSparseTensor(tnsSparseTensor *tsr, FILE *fp)
{
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
    
    tnsNewSparseTensor(tsr, ndims_temp, nmodes_temp, nnz);
    free(ndims_temp);
    //为nnz开辟val和index空间
    for(mode=0; mode < tsr->nmodes; ++mode){
        tnsNewIndexVector(&tsr->inds[mode], tsr->nnz);
    }
    tnsNewValueVector(&tsr->values, tsr->nnz);
    tnsIndex nnz_chk = 0;
    char *ptr = NULL;
    fseek(fp, fp_local, 0);
    while((getline(&line, &len, fp)) != -1)
    {
        ptr = line;
        for(mode = 0; mode < tsr->nmodes; ++mode)
        {
            tsr->inds[mode].values[nnz_chk] = strtoul(ptr, &ptr, 10);
        }
        tsr->values.values[nnz_chk] = strtod(ptr, &ptr);
        ++nnz_chk;
    }

    if(nnz != nnz_chk)
    {
        printf("getline函数出错\n");
        printf("nnz=%d nnz_chk=%d\n", nnz, nnz_chk);    
    }
    
    return 0;
}





void tnsFreeSparseTensor(tnsSparseTensor *tsr){
    free(tsr->ndims);
    tnsFreeValueVector(&tsr->values);
    
    for(tnsIndex i = 0; i < tsr->nmodes; ++i){
        tnsFreeIndexVector(&tsr->inds[i]);
    }
    tsr->nmodes = 0;
    tsr->nnz = 0;
    free(tsr->inds);
}