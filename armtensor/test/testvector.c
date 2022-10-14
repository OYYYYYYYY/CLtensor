#include "TArm.h"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int main(void) {
    

    tnsIndexVector vic;
    tnsIndexVector cpvic;
    tnsValueVector vvc; 
    tnsValueVector cpvvc;
    FILE *fp;
    tnsIndex len = 10;
    /* int a[10] = {0,0,0,0,0,0,0,0,0,0};
    #pragma omp parallel for
    for(int i = 0; i < 10; i++){
        // a[i] = 1;
        printf("%d ", i);
    } */
    tnsIndex ivalue = 5;
    tnsValue vvalue = 6;
    
    //tns_CheckOSError(0 , "tns New"); 
    //tns_CheckError(1 , "sptMatrixMulMatrix", "维度有问题！");
    
    //初始化向量
    tnsNewIndexVector(&vic, len);
    tnsNewIndexVector(&cpvic, len);
    tnsNewValueVector(&vvc, len);
    tnsNewValueVector(&cpvvc, len);
    //赋常数值
    tnsConstantIndexVector(&vic, ivalue);
    tnsConstantValueVector(&vvc, vvalue);
    //拷贝向量
    tnsCopyIndexVector(&cpvic, &vic);
    tnsCopyValueVector(&cpvvc, &vvc);
    //从文件读入IndexVector
    fp = fopen("./data/test_vector.txt", "r");
    tnsLoadIndexVector(&vic, fp);
    //IndexVector输出到文件
    fp = fopen("./data/result.txt", "w");
    tnsDumpIndexVector(&vic, fp);
    //从文件读入ValueVector
    fp = fopen("./data/test_vector.txt", "r");
    tnsLoadValueVector(&vvc, fp);
    //ValueVector输出到文件
    fp = fopen("./data/result.txt", "w");
    tnsDumpValueVector(&vvc, fp);
    //添加元素
    tnsAppendIndexVector(&vic, ivalue);
    tnsAppendValueVector(&vvc, vvalue);
    for(tnsIndex i = 0; i < vvc.nlens; ++i)
        printf("%f ", vvc.values[i]);
    printf("\n");
    
    //释放向量
    tnsFreeIndexVector(&vic);
    tnsFreeIndexVector(&cpvic);
    tnsFreeValueVector(&vvc);
    tnsFreeValueVector(&cpvvc);
    free(fp);
    printf("finish\n");
    

    return 0;
}