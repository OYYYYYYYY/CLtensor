# 声明CMake的最低要求版本
cmake_minimum_required(VERSION 3.2)
# 定义项目(工程)名称,
# 同时定义了以下默认变量:
# PROJECT_SOURCE_DIR 或 HELLO_SOURCE_DIR, 代指CMake开始构建的根目录(通常是项目根目录)
# PROJECT_BINARY_DIR 或 HELLO_BINARY_DIR, 代指CMake的编译目录(即执行cmake命令的目录)
PROJECT(TArmProject)



# Define data types
add_definitions(-DTARM_INDEX_TYPEWIDTH=32)
add_definitions(-DTARM_VALUE_TYPEWIDTH=32)
# 添加参与编译的子目录
# src写功能
ADD_SUBDIRECTORY(src)
# test写测试
ADD_SUBDIRECTORY(test)


