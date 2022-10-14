# 简介
ARM架构下 张量库 的实现。
稀疏格式统一采用COO储存。


---
其中FT 2000+ 中build.sh可能无法直接用，则直接在bash中执行命令
```
mkdir -p build
cd build
cmake ..
make
```
---
基础算子

- SpTTM
```
./build/test/test_ttmmix ./data/uber.tns ./data/matrix_500_1140.txt 2
```



--- 
目录结构：示例
```
<!-- 头文件位置 -->
include/
  TArm/
    structs.h
    tensor.h
  TArm.h

<!-- 功能文件位置 -->
src/
  tensor/
    tensor.c

<!-- 测试文件位置(包括main) -->
test/
  testvector.c



```

