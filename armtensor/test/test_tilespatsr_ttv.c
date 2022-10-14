
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
    int iteration = 5;

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
    printf("load tensor success! \n");

    tnsValueVector vec_mul;
    tnsNewValueVector(&vec_mul, tiletsr.ndims[compute_mode]);
    tnsConstantValueVector(&vec_mul, 0.1);

    tnsIndexVector order;
    tnsNewIndexVector(&order, tiletsr.nmodes);
    tnsContinuousIndexVector(&order, 0, 1);
    order.values[tiletsr.nmodes-1] = compute_mode;
    order.values[compute_mode] = tiletsr.nmodes-1;

    printf("order success! \n");
    // tnsPermuteTileSpatsr之后compute_mode 置为tiletsr.nmodes-1
    tnsPermuteTileSpatsr(&tiletsr, order.values);
    compute_mode = tiletsr.nmodes-1;
    printf("tnsPermuteTileSpatsr success! \n");
    
    // 结果张量
    tnsSparseTensor Y_tsr;
    tnsIndex *Y_ndims;
    Y_ndims = calloc(tiletsr.nmodes-1, sizeof *Y_ndims);
    tnsIndex y_mode_i = 0;
    for(int mode_i = 0; mode_i < tiletsr.nmodes; ++mode_i){
        if(mode_i != compute_mode){
            Y_ndims[y_mode_i] = tiletsr.ndims[mode_i];
            ++y_mode_i;
        }///< if
    }///< for mode_i
    printf("Y_ndims success! \n");

    tnsNewSparseTensor(&Y_tsr, Y_ndims, tiletsr.nmodes-1, 0);
    tnsVecTilingTileSpatsr(&Y_tsr, &tiletsr);
    printf("tnsVecTilingTileSpatsr success! \n");

    // 开始测试
    // 大数组的 openmp reduction 循环会段错误
    if(tiletsr.nnz_ndims[tiletsr.nmodes-1] > 1000){
        iteration = 1;
    }
    timer_restart(&test_timer);
    for(int iter_i = 0; iter_i < iteration; ++iter_i){
        tnsTTVTileSpatsr(&Y_tsr, &tiletsr, &vec_mul, tiletsr.nmodes-1, tk);
    }
    timer_stop(&test_timer);
    timer_print_iter_sec(&test_timer, iteration,  "tnsNaiveMTTKRPTileSpatsr2");

    tnsFreeValueVector(&vec_mul);

    // printf("nnz:%d \n",tiletsr.nnz);

    // tnsFreeSparseTensor(&tsr);
    free(Y_ndims);
    tnsFreeTileSpatsr(&tiletsr);
    fclose(fi);

    return 0;

}