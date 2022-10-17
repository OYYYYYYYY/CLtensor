#include "TArm.h"
#include <stdlib.h>
#include <string.h>



// #IndexVector# //
// IndexVector分配空间
int tnsNewIndexVector(tnsIndexVector *vec, tnsIndex len) {
    vec->nlens = len;
    vec->memory = len;
    vec->values = malloc(vec->nlens * sizeof *vec->values);
    memset(vec->values, 0, len * sizeof *vec->values);
    return 0;
}

// IndexVector释放空间
void tnsFreeIndexVector(tnsIndexVector *vec) {
    vec->nlens = 0;
    vec->memory = 0;
    free(vec->values);
    vec->values = NULL;  
}

// IndexVector赋常数值
int tnsConstantIndexVector(tnsIndexVector * const vec, tnsIndex const val) {
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        vec->values[i] = val;
    return 0;
}


// 根据初始值赋连续数值
int tnsContinuousIndexVector(tnsIndexVector * const vec, tnsIndex val, tnsIndex step) {
    tnsIndex i;
    #pragma omp parallel for num_threads(32)
    for(i = 0; i < vec->nlens; ++i){
        vec->values[i] = val + i*step;;
    }
        
    return 0;
}

// IndexVector拷贝vector
int tnsCopyIndexVector(tnsIndexVector *dest, const tnsIndexVector *src) {
    if(dest->nlens != src->nlens)
        return 1;
    memcpy(dest->values, src->values, dest->nlens * sizeof *dest->values);
    return 0;
}

// IndexVector拷贝vector
int tnsPartCopyIndexVector(tnsIndexVector *dest, const tnsIndexVector *src, tnsIndex len) {
    if(dest->nlens < src->nlens)
        return 1;
    memcpy(dest->values, src->values, len * sizeof *dest->values);
    return 0;
}
// IndexVector从文件读入
int tnsLoadIndexVector(tnsIndexVector *vec, FILE *fp) {
    fscanf(fp, "%u", &vec->nlens);
    
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        fscanf(fp, "%u", &vec->values[i]);
    return 0;
}

// IndexVector存入文件
int tnsDumpIndexVector(tnsIndexVector *vec, FILE *fp) {
    fprintf(fp, "%u\n", vec->nlens);
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        fprintf(fp, "%u ", vec->values[i]);
    return 0;
} 

// IndexVector添加元素
int tnsAppendIndexVector(tnsIndexVector *vec, tnsValue const val) {
    // 大于当前开辟的空间，增加100的的数目。
    if((vec->nlens + 1) > vec->memory){
        vec->memory += 10;
        tnsIndex *newvalues = realloc(vec->values, (vec->nlens + 10) * sizeof *vec->values);
        vec->values = newvalues;
        newvalues = NULL;
    }
    // tnsIndex *newvalues = realloc(vec->values, (vec->nlens + 1) * sizeof *vec->values);
    // vec->values = newvalues;
    vec->values[vec->nlens] = val;
    ++vec->nlens;
    return 0;
}

/// 操作函数
/// 原数组映射交换 形成新的数组。
int tnsMapSwapIndexVector(tnsIndexVector * dest, tnsIndexVector * src){
    // tns_CheckError((src->nlens != dest->nlens),"tnsMapSwapIndexVector","数组大小不同!")
    tnsIndex i;
    #pragma omp parallel for num_threads(32)
    for(i = 0; i < src->nlens; ++i){
        dest->values[src->values[i]] = i;
    }
    return 0;
}



// #ValueVector# //
// ValueVector分配空间
int tnsNewValueVector(tnsValueVector *vec, tnsIndex len) {
    vec->nlens = len;
    vec->values = malloc(len * sizeof *vec->values);
    memset(vec->values, 0, len * sizeof *vec->values);
    return 0;
}

// ValueVector释放空间
void tnsFreeValueVector(tnsValueVector *vec) {
    vec->nlens = 0;  
    free(vec->values);
    vec->values = NULL;  
}

// ValueVector赋常数值
int tnsConstantValueVector(tnsValueVector * const vec, tnsValue const val) {
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        vec->values[i] = val;
    return 0;
}

// ValueVector拷贝vector
int tnsCopyValueVector(tnsValueVector *dest, const tnsValueVector *src) {
    if(dest->nlens != src->nlens)
        return 1;
    memcpy(dest->values, src->values, dest->nlens * sizeof *dest->values);
    return 0;
}

// ValueVector从文件读入
int tnsLoadValueVector(tnsValueVector *vec, FILE *fp) {
    fscanf(fp, "%u", &vec->nlens);
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        fscanf(fp, "%f", &vec->values[i]);
    return 0;
}

// ValueVector存入文件
int tnsDumpValueVector(tnsValueVector *vec, FILE *fp) {
    fprintf(fp, "%u\n", vec->nlens);
    for(tnsIndex i = 0; i < vec->nlens; ++i)
        fprintf(fp, "%f ", vec->values[i]);
    return 0;
}

// ValueVector添加元素
int tnsAppendValueVector(tnsValueVector *vec, tnsValue const val) {
    // 大于当前开辟的空间，增加100的的数目。
    if((vec->nlens + 1) > vec->memory){
        vec->memory += 10;
        tnsIndex *newvalues = realloc(vec->values, (vec->nlens + 10) * sizeof *vec->values);
        vec->values = newvalues;
        newvalues = NULL;
    }
    vec->values[vec->nlens] = val;
    ++vec->nlens;
    // free(newvalues);
    return 0;
}