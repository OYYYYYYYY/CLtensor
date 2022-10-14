#include <TArm.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clapack.h"


/* mats (aTa) only stores upper triangle elements. */
int tnsDenseMatrixDotMulSeqTriangle(tnsIndex const mode, tnsIndex const nmodes, tnsDenseMatrix ** mats)
{
    tnsIndex const nrows = mats[0]->nrows;
    tnsIndex const ncols = mats[0]->ncols;
    tnsIndex const stride = mats[0]->stride; 
    // for(tnsIndex m=1; m<nmodes+1; ++m) {
    //     assert(mats[m]->ncols == ncols);
    //     assert(mats[m]->nrows == nrows);
    // }

    tnsValue * ovals = mats[nmodes]->values.values;
    #pragma omp parallel for schedule(static)
    for(tnsIndex i=0; i < nrows; ++i) {
        for(tnsIndex j=0; j < ncols; ++j) {
            ovals[j * stride + i] = 1.0;
        }
    }


    for(tnsIndex m=1; m < nmodes; ++m) {
        tnsIndex const pm = (mode + m) % nmodes;
        tnsValue const * vals = mats[pm]->values.values;
    #pragma omp parallel for schedule(static)
        for(tnsIndex i=0; i < nrows; ++i) {
            for(tnsIndex j=i; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }

    /* Copy upper triangle to lower part */
    #pragma omp parallel for schedule(static)
    for(tnsIndex i=0; i < nrows; ++i) {
        for(tnsIndex j=0; j < i; ++j) {
            ovals[i * stride + j] = ovals[j * stride + i];
        }
    }
    
    return 0;
}


// Row-major
int tnsDenseMatrix2Norm(tnsDenseMatrix * const A, tnsValue * const lambda)
{
    tnsIndex const nrows = A->nrows;
    tnsIndex const ncols = A->ncols;
    tnsIndex const stride = A->stride;
    tnsValue * const vals = A->values.values;
    tnsValue * buffer_lambda;

    #pragma omp parallel for
    for(tnsIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (tnsValue *)malloc(nthreads * ncols * sizeof(tnsValue));
            for(tnsNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        tnsValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(tnsIndex i=0; i < nrows; ++i) {
            for(tnsIndex j=0; j < ncols; ++j) {
                loc_lambda[j] += vals[i*stride + j] * vals[i*stride + j];
            }
        }

        #pragma omp for
        for(tnsIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                lambda[j] += buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */


        #pragma omp parallel for
        for(tnsIndex j=0; j < ncols; ++j) {
            lambda[j] = sqrt(lambda[j]);
        }

        #pragma omp parallel for
        for(tnsIndex i=0; i < nrows; ++i) {
            for(tnsIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

    
    free(buffer_lambda);

    return 0;
}

// Row-major
int tnsDenseMatrixMaxNorm(tnsDenseMatrix * const A, tnsValue * const lambda)
{
    tnsIndex const nrows = A->nrows;
    tnsIndex const ncols = A->ncols;
    tnsIndex const stride = A->stride;
    tnsValue * const vals = A->values.values;
    tnsValue * buffer_lambda;

    #pragma omp parallel for
    for(tnsIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (tnsValue *)malloc(nthreads * ncols * sizeof(tnsValue));
            for(tnsNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        tnsValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(tnsIndex i=0; i < nrows; ++i) {
            for(tnsIndex j=0; j < ncols; ++j) {
                if(vals[i*stride + j] > loc_lambda[j])
                    loc_lambda[j] = vals[i*stride + j];
            }
        }

        #pragma omp for
        for(tnsIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                if(buffer_lambda[i*ncols + j] > lambda[j])
                    lambda[j] = buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */


        #pragma omp parallel for
        for(tnsIndex j=0; j < ncols; ++j) {
            if(lambda[j] < 1)
                lambda[j] = 1;
        }

        #pragma omp parallel for
        for(tnsIndex i=0; i < nrows; ++i) {
            for(tnsIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

    free(buffer_lambda);

    return 0;
}

int tnsDenseMatrixSolveNormals(
  tnsIndex const mode,
  tnsIndex const nmodes,
  tnsDenseMatrix ** aTa,
  tnsDenseMatrix * rhs)
{
  int rank = (int)(aTa[0]->ncols);
  int stride = (int)(aTa[0]->stride);

  tnsDenseMatrixDotMulSeqTriangle(mode, nmodes, aTa);

  int info;
  char uplo = 'L';
  int nrhs = (int) rhs->nrows;
  tnsValue * const neqs = aTa[nmodes]->values.values;

  /* Cholesky factorization */
  int is_spd = 1;
  // lapackf77_spotrf(&uplo, &rank, neqs, &stride, &info);
  spotrf_(&uplo, &rank, neqs, &stride, &info);
  if(info) {
    // printf("Gram matrix is not SPD. Trying `gesv`.\n");
    is_spd = 0;
  }

  /* Continue with Cholesky */
  if(is_spd) {
    /* Solve against rhs */
    // lapackf77_spotrs(&uplo, &rank, &nrhs, neqs, &stride, rhs->values, &stride, &info);
    spotrs_(&uplo, &rank, &nrhs, neqs, &stride, rhs->values.values, &stride, &info);
    if(info) {
    //   printf("DPOTRS returned %d\n", info);
    }
  } 
  else {
    int * ipiv = (int*)malloc(rank * sizeof(int));  

    /* restore gram matrix */
    tnsDenseMatrixDotMulSeqTriangle(mode, nmodes, aTa);

    sgesv_(&rank, &nrhs, neqs, &stride, ipiv, rhs->values.values, &stride, &info);
    // lapackf77_sgesv(&rank, &nrhs, neqs, &stride, ipiv, rhs->values, &stride, &info);
    // magma_sgesv(rank, nrhs, neqs, stride, ipiv, rhs->values, stride, &info);
    if(info) {
    //   printf("sgesv_ returned %d\n", info);
    }

    free(ipiv);
  }

  return 0;
}