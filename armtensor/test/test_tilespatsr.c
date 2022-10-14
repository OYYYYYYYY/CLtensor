
#include <TArm.h>
#include <timer.h>

#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
// #include <string.h>
#include "omp.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes, or specify a mode, e.g., 0 or 1 or 2 for third-order tensors.)\n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         --help\n");
    printf("\n");
}
int main(int argc, char ** argv) {
    int tk = 1;
    int iteration = 1;

    tnsIndex compute_mode = 9;
    // 输入参数处理
    if(argc <= 1) { // #Required arguments
        print_usage(argv);
        exit(1);
    }
    int c;
    FILE* fi;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"mode", required_argument, 0, 'm'},
            {"nt", optional_argument, 0, 't'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            // tnsAssert(fi != NULL);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%d", &compute_mode);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    Timer test_timer;

    tnsSparseTensor tsr;
    // FILE* fp_r = fopen("data/whtdata/10_100000.tns", "r");
    // FILE* fp_r = fopen("data/whtdata/ten_5_5_5_5.tns", "r");
    // tnsLoadSparseTensor(&tsr, fp_r);

    tnsTileSpatsr tiletsr;
    tnsLoadTileSpatsr(&tiletsr, fi);
    printf("从文件中读取稀疏张量 success\n");

    char mat_name[255];
    tnsDenseMatrix **U_list =(tnsDenseMatrix **)malloc((tiletsr.nmodes+1) * sizeof(*U_list));
    int rank = 8;
    tnsIndex max_dim_value = 0;
    int nmodes = tiletsr.nmodes;
    
    // 初始化因子矩阵
    for(tnsIndex mode_i = 0; mode_i < tiletsr.nmodes; ++mode_i){
        // 类似于申明 tnsDenseMatrix U_list[mode_i] 的作用
        U_list[mode_i] = (tnsDenseMatrix *)malloc(sizeof(tnsDenseMatrix));
        tnsNewDenseMatrix(U_list[mode_i], tiletsr.ndims[mode_i], rank);
        tnsConstantDenseMatrix(U_list[mode_i], 1);
        // tnsRandomDenseMatrix(&U_list[mode_i]);
        if(max_dim_value < tiletsr.ndims[mode_i]){
            max_dim_value = tiletsr.ndims[mode_i];
        }
        
    }
    // 初始化因子矩阵的临时矩阵
    U_list[nmodes] = (tnsDenseMatrix *)malloc(sizeof(tnsDenseMatrix));
    tnsNewDenseMatrix(U_list[nmodes], max_dim_value, rank);
    // 暂时用于测试，应该设置为0
    tnsConstantDenseMatrix(U_list[nmodes], 1);
    
    // 根据computer_mode调整 U 的顺序，使维度对应。
    // tnsDenseMatrix *compute_U_list = malloc(tiletsr.nmodes * sizeof *compute_U_list);
    // compute_U_list[0] = U_list[compute_mode];
    // for(tnsIndex ui = 1; ui < tiletsr.nmodes; ++ui){
    //     if(ui == compute_mode){
    //         compute_U_list[compute_mode] = U_list[0];
    //     }else{
    //         compute_U_list[ui] = U_list[ui];
    //     }
    // }
    // printf("load compute_U_list success\n");
    // timer_start(&test_timer);
    // for(int iter_i = 0; iter_i < iteration;++iter_i){
    //     // 缺少分块属性
    //     // tnsNaiveMTTKRPTileSpatsr(&tiletsr, compute_U_list, compute_mode, tk);
    // }
    // timer_stop(&test_timer);
    // timer_print_iter_sec(&test_timer, iteration,  "tnsNaiveMTTKRPTileSpatsr");


    // tnsIndex order[4]={2,3,0,1};
    // tnsIndex order[4]={0,1,2,3};
    // tnsPermuteTileSpatsr(&tiletsr, order);
    tnsRedimTileSpatsr(&tiletsr);
    printf("tnsRedimTileSpatsr success\n");
    // tnsPermuteTileSpatsr(&tiletsr, order);
    // tnsNewTileSpatsr(&tiletsr, tsr.ndims, tsr.nmodes, tsr.nnz);
    // tnsSparseTensor2TileSpatsr(&tiletsr, &tsr);
    // tnsIndex block_size[4]={10,10,10,10};
    tnsIndex block_size[10]={3,3,3,3,3,3,3,3,3,1000};
    tnsIndex sp_threshold = 100;
    tnsTilingTileSpatsr(&tiletsr, block_size, sp_threshold);
    printf("tnsTilingTileSpatsr success\n");
    tnsRetilingTileSpatsr(&tiletsr, compute_mode);
    printf("tnsRetilingTileSpatsr success\n");
    // FILE* fp_w = fopen("../data/whtdata/res.tns", "w");
    // tnsDumpTileSpatsr(&tiletsr, fp_w);
    // fclose(fp_w);
    
    // 开始测试没有分块,有排序
    // for(tnsIndex mode_i = 0; mode_i < tiletsr.nmodes; ++mode_i){
    //     tnsNewDenseMatrix(&U_list[mode_i], tiletsr.ndims[mode_i], rank);
    //     tnsRandomDenseMatrix(&U_list[mode_i]);
    //     // sprintf(mat_name, "data/whtdata/mat_%d00_32.mtx", (mode_i+1));
    //     // fp_r = fopen(mat_name, "r");
    //     // tnsLoadDenseMatrix(&U_list[mode_i], fp_r);
    // }
    // 根据computer_mode调整 U 的顺序，使维度对应。
    // compute_U_list = malloc(tiletsr.nmodes * sizeof *compute_U_list);
    // compute_U_list[0] = U_list[compute_mode];
    // for(tnsIndex ui = 1; ui < tiletsr.nmodes; ++ui){
    //     if(ui == compute_mode){
    //         compute_U_list[compute_mode] = U_list[0];
    //     }else{
    //         compute_U_list[ui] = U_list[ui];
    //     }
    // }
    // printf("load compute_U_list success\n");
    // 大数组的 openmp reduction 循环会段错误
    if(tiletsr.nnz_ndims[compute_mode] > 1000){
        iteration = 1;
    }
    timer_restart(&test_timer);
    for(int iter_i = 0; iter_i < iteration;++iter_i){
        tnsNaiveMTTKRPTileSpatsr(&tiletsr, U_list, compute_mode, tk);
    }
    timer_stop(&test_timer);
    timer_print_iter_sec(&test_timer, iteration,  "tnsNaiveMTTKRPTileSpatsr2");
    
    // FILE * fp_res_w = fopen("../data/whtdata/result.mat", "w");
    // tnsDumpDenseMatrix(U_list[compute_mode], fp_res_w);
    // fclose(fp_res_w);

    // 开始计算

    //     sprintf(outname, "/data/whtdata/projects/ParTI-master/tensors/mat1nell/%d.txt", tm);
    //     fp=fopen(outname,"w");

    // for(tnsIndex mode_i = 0; mode_i < tiletsr.nmodes; ++mode_i){
    //     tnsNewDenseMatrix(&U_list[mode_i], tiletsr.ndims[mode_i], rank);
    //     tnsRandomDenseMatrix(&U_list[mode_i]);
    //     // sprintf(mat_name, "data/whtdata/mat_%d00_32.mtx", (mode_i+1));
    //     // fp_r = fopen(mat_name, "r");
    //     // tnsLoadDenseMatrix(&U_list[mode_i], fp_r);
    // }

    // 根据computer_mode调整 U 的顺序，使维度对应。
    // compute_U_list = malloc(tiletsr.nmodes * sizeof *compute_U_list);
    // compute_U_list[0] = U_list[compute_mode];
    // for(tnsIndex ui = 1; ui < tiletsr.nmodes; ++ui){
    //     if(ui == compute_mode){
    //         compute_U_list[compute_mode] = U_list[0];
    //     }else{
    //         compute_U_list[ui] = U_list[ui];
    //     }
    // }
    // printf("load compute_U_list success\n");
    // // 大数组的 openmp reduction 循环会段错误
    // if(tiletsr.nnz_ndims[compute_mode] > 1000){
    //     iteration = 1;
    // }
    // // 看开始计算OMP
    // timer_restart(&test_timer);for(int iter_i = 0; iter_i < iteration;++iter_i){
    //     tnsOmpMTTKRPTileSpatsr(&tiletsr, compute_U_list, compute_mode, tk);
    // }
    // timer_stop(&test_timer);
    // timer_print_iter_sec(&test_timer, iteration,  "tnsOmpMTTKRPTileSpatsr");
    
    



// 开始测试单线程
    // for(tnsIndex mode_i = 0; mode_i < tiletsr.nmodes; ++mode_i){
    //     sprintf(mat_name, "data/whtdata/mat_%d00_32.mtx", (mode_i+1));
    //     // sprintf(mat_name, "data/whtdata/dense_5_5.mtx");
    //     fp_r = fopen(mat_name, "r");
    //     tnsLoadDenseMatrix(&U_list[mode_i], fp_r);
    // }
    // // 根据computer_mode调整 U 的顺序，使维度对应。
    // compute_U_list = malloc(tiletsr.nmodes * sizeof *compute_U_list);
    // compute_U_list[0] = U_list[compute_mode];
    // for(tnsIndex ui = 1; ui < tiletsr.nmodes; ++ui){
    //     if(ui == compute_mode){
    //         compute_U_list[compute_mode] = U_list[0];
    //     }else{
    //         compute_U_list[ui] = U_list[ui];
    //     }
    // }
    // printf("load compute_U_list success\n");
    // timer_start(&test_timer);
    // tnsMTTKRPTileSpatsr(&tiletsr, compute_U_list);
    // // tnsOmpMTTKRPTileSpatsr(&tiletsr, compute_U_list, 32);
    // timer_stop(&test_timer);
    // timer_print_sec(&test_timer, "tnsMTTKRPTileSpatsr");


    // FILE * fp_res_w = fopen("../data/whtdata/result.mat", "w");
    // tnsDumpDenseMatrix(&U_list[compute_mode], fp_res_w);
    // fclose(fp_res_w);



    for(tnsIndex ui = 0; ui < tiletsr.nmodes; ++ui){
        tnsFreeDenseMatrix(U_list[ui]);
    }
    free(U_list);
    // free(compute_U_list);

    // printf("nnz:%d \n",tiletsr.nnz);

    // tnsFreeSparseTensor(&tsr);
    tnsFreeTileSpatsr(&tiletsr);
    fclose(fi);

    return 0;

}