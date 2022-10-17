#include "timer.h"
#include <stdio.h>

void timer_reset(Timer * const kTimer) {
  kTimer->running       = 0;
  kTimer->seconds       = 0;
  kTimer->u_seconds     = 0;
  kTimer->Start.tv_sec  = 0;
  kTimer->Start.tv_usec = 0;
  kTimer->Stop.tv_sec   = 0;
  kTimer->Stop.tv_usec  = 0;
}


void timer_start(Timer * const kTimer) {
  kTimer->running = 1;
  gettimeofday(&(kTimer->Start), NULL);
}


void timer_stop(Timer * const kTimer) {
  kTimer->running = 0;
  gettimeofday(&(kTimer->Stop), NULL);
  /// 统计秒为单位
  kTimer->seconds += (double)(kTimer->Stop.tv_sec - kTimer->Start.tv_sec);
  kTimer->seconds += 1e-6 * (kTimer->Stop.tv_usec - kTimer->Start.tv_usec);
  
  /// 重新统计微秒为单位
  kTimer->u_seconds += 1e6 * (double)(kTimer->Stop.tv_sec - kTimer->Start.tv_sec);
  kTimer->u_seconds += (kTimer->Stop.tv_usec - kTimer->Start.tv_usec);

}


void timer_restart(Timer * const kTimer) {
  timer_reset(kTimer);
  timer_start(kTimer);
}


void timer_print_sec(Timer * const kTimer, const char *remark){
    printf("'%s' module costtime is : %lf s\n", remark, kTimer->seconds);
}

void timer_print_usec(Timer * const kTimer, const char *remark){
    printf("'%s' module costtime is : %lf us\n", remark, kTimer->u_seconds);
}

void timer_print_iter_sec(Timer * const kTimer, int iters, const char *remark){
    printf("'%s' module %d iters avg time is : %lf s\n", remark, iters, kTimer->seconds/iters);
}


