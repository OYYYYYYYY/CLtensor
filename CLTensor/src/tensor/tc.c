#include "omp.h"
#include <TArm.h>
#include <stdlib.h>
#include <string.h>



/**
  * @fn tensor contraction between two tensors
  * @brief 根据contract mode计算两个张量的乘积
  * @attention C由外部开辟空间
  * @param[in] A 输入张量
  * @param[in] B 输入张量
  * @param[in] I_n 张量A的contract mode
  * @param[in] J_m 张量B的contract mode
  * @param[in] con_mode contract mode的数量
  * @param[out] C 输出张量
  * @return 返回函数正确
  
  */
int tnsTCSpatsr(tnsSparseTensor *C, tnsSparseTensor *A, tnsSparseTensor *B, tnsIndex *I_n, tnsIndex *J_m, tnsIndex con_num){

  for(tnsIndex i = 0; i < con_num; ++i)
  {
    tns_CheckOSError(A->ndims[I_n[i]] != B->ndims[J_m[i]], "contract mode 维度不匹配");
  }
  

  return 0;
}




