## 命名统一
- 暴露的函数前缀：tns
- 稀疏矩阵简写：Spamat
- 稠密矩阵简写：Denmat
- 稀疏张量简写：Spatsr
- 稠密张量简写：Dentsr
- ×××函数，需要另开空间保存结果。
- ×××_rw函数，覆盖原空间保存结果。

- 计算的mode：copt_mode
- 索引的mode：indx_mode

- 线程的参数名：tk

## 注释规范
- 多行注释
```
/**
* 注释内容
*/
```
- 单行注释
```
/// 注释内容
```
- 行后注释
```
///< 注释内容
```
- 函数注释
```
/**
  * @fn 函数名
  * @brief 简述函数功能。
  * @details 提示一些注意事项或必要的技术细节。
  * @param[in] 参数名 参数注解
  * @param[out] 参数名 参数注解
  * @return 描述返回意义
  * @retval 描述返回值意义
  * @note 注解。
  * @attention 注意事项。
  * @par example:
  * @code
  //代码示例
  * @endcode
  */
举例
/**
  * @fn tensor contraction between two tensors
  * @brief 根据contract mode计算两个张量的乘积
  * 
  * @param[in] A 输入张量
  * @param[in] B 输入张量
  * @param[in] I_n 张量A的contract mode
  * @param[in] J_m 张量B的contract mode
  * @param[in] con_mode contract mode的数量
  * @param[out] C 输出张量
  * @return 返回函数正确
  
  */

```


## 错误处理
- tns_CheckOSError(cond, module) :检查函数传入参数的错误,使用此接口.
- tns_CheckError(errcode, module, reason) :检查单线程计算时的错误,使用此接口. 
- tns_CheckOmpError(errcode, module, reason) :检查多线程计算时的错误,使用此接口 
```
/// cond 为1（也是不为0），则报错 / 判断条件为真，则报错，并终止程序。
tns_CheckOSError(len, "tens New");
/// errcode 不为0，则报错，并终止程序 / 判断条件为真，则报错，并终止程序
tns_CheckError(len, "tens New");
tns_CheckOmpError(len, "tens New");


```

## 程序计时
示例如下：
```
/// 结构体声明
Timer test_timer;
/// 开始计时
timer_start(&test_timer);

/**
* 需要计时的函数
*/

/// 结束计时
timer_stop(&test_timer);
/// 输出的结果以 秒 为单位（二选一）。
timer_print_sec(&test_timer, "test_func");

/// 可重复使用结构体restart
timer_restart(&test_timer);

/**
* 需要计时的函数
*/

timer_stop(&test_timer);
/// 输出的结果以 微秒 为单位（二选一）。
timer_print_usec(&test_timer, "test_func");

```

## 注意事项
- 张量开辟空间的函数包含以下，其余函数使用前需要在函数外部开辟空间
  1. tnsLoadSparseTensor
  2. tnsNewSparseTensor
  3. tnsLoadDenseTensor
  4. tnsNewDenseTensor
