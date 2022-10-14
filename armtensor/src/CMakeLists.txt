
option(USE_OPENMP "Use OPENMP" ON)

if(USE_OPENMP)
    add_definitions(-DTARM_USE_OPENMP)
    if(USE_ICC)
        if(DEFINED DEBUG)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -qopenmp -O0 -g -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -qopenmp -O0 -g -DADD_")
        else()
            if(USE_KNL)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmic -std=gnu99 -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -qopenmp -O3 -DADD_")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic -Wall -Wextra -Wno-maybe-uninitialized -lrt -qopenmp -O3 -DADD_")
            else()
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -qopenmp -O3 -DADD_")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -qopenmp -O3 -DADD_")
            endif()
        endif()
    else()
        if(DEFINED DEBUG)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -fopenmp -O0 -g -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -fopenmp -O0 -g -DADD_")
        else()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -fopenmp -O3 -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -fopenmp -O3 -DADD_")
        endif()
    endif()
else()
    if(DEFINED DEBUG)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -O0 -g -DADD_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -O0 -g -DADD_")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -O3 -DADD_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -O3 -DADD_")
    endif()
endif()

# 添加头文件的查找目录
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)

# 指定库文件输出路径
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE TARM_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*/*.c" "*.c")
# file(GLOB_RECURSE TARM_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/*/*.c" "${PROJECT_SOURCE_DIR}/include/*.h")
# 查找所有源文件，并将名称保存到 DIR_LIB_SRCS 变量
# aux_source_directory("${CMAKE_CURRENT_SOURCE_DIR}" DIR_LIB_SRCS)

# 添加数学库
ADD_LIBRARY (TArm_so SHARED  ${TARM_SRC})
SET_TARGET_PROPERTIES(TArm_so PROPERTIES OUTPUT_NAME "TArm")

