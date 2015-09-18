#include <stdio.h>
#include "cblas.h"
#include "omp.h"
#include <time.h>
#include <stdlib.h>
#include "qute.h"

void method1(int M,int N,int T,double *Xin,double *Xout,double *Kern) {
    for (int n=0; n<N-T; n++) {
        for (int m=0; m<M; m++) {
            double val=0;
            for (int dt=0; dt<T; dt++) {
                val+=Xin[m+M*(n+dt)]*Kern[dt];
            }
            Xout[m+M*n]=val;
        }
    }
}

double dotprod1(int N,double *X,int incx,double *Y,int incy) {
    double ret=0;
    int ix=0,iy=0;
    for (int i=0; i<N; ++i) {
        ret+=X[ix]*Y[iy];
        ix+=incx; iy+=incy;
    }
    return ret;
}

void method2(int M,int N,int T,double *Xin,double *Xout,double *Kern) {
    for (int n=0; n<N-T; n++) {
        for (int m=0; m<M; m++) {
            double val=dotprod1(T,&Xin[m+M*n],M,Kern,1);
            Xout[m+M*n]=val;
        }
    }
}

double dotprod2(int N,double *X,int incx,double *Y,int incy) {
    return cblas_ddot(N, X, incx, Y, incy);
}

void method3(int M,int N,int T,double *Xin,double *Xout,double *Kern) {
    for (int n=0; n<N-T; n++) {
        for (int m=0; m<M; m++) {
            double val=dotprod2(T,&Xin[m+M*n],M,Kern,1);
            Xout[m+M*n]=val;
        }
    }
}

void method4(int M,int N,int T,double *Xin,double *Xout,double *Kern) {
    for (int n=0; n<N-T; n++) {
        cblas_dgemv(CblasColMajor, CblasNoTrans, M, T, 1.0, &Xin[M*n], M, Kern, 1, 0.0, &Xout[M*n], 1);
    }
}

void method5(int M,int N,int T,double *Xin,double *Xout,double *Kern) {

    omp_set_num_threads(10);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads=omp_get_num_threads();

    if (tid==0) printf("Using %d threads\n",omp_get_num_threads());

    int i1=((N-T)*tid)/nthreads;
    int i2=((N-T)*(tid+1))/nthreads;
    for (int n=i1; n<i2; n++) {
        cblas_dgemv(CblasColMajor, CblasNoTrans, M, T, 1.0, &Xin[M*n], M, Kern, 1, 0.0, &Xout[M*n], 1);
    }
  }
}


int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    srand(time(NULL));

    long M=512; //number of channels
    long T=100; //convolution kernel size
    long N=5e9/(M*T); //number of timepoints

    double *Xin=(double *)malloc(sizeof(double)*M*N);
    double *Xout=(double *)malloc(sizeof(double)*M*N);
    double *Kern=(double *)malloc(sizeof(double)*T);

    printf("Note: Each operation consists of a multiply and an add.\n\n");

    printf("Preparing...\n");
    for (int ii=0; ii<M*N; ii++) {
        Xin[ii]=ii%183;
    }
    for (int t=0; t<T; t++) {
        Kern[t]=rand()*1.0/rand();
    }

    if (1) {
        printf("\nMethod 1 (simple loops)...\n");
        QTime timer; timer.start();
        method1(M,N,T,Xin,Xout,Kern);
        double elapsed=timer.elapsed()*1.0/1000;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        double mops=M*N*T/elapsed/1e6;
        printf("Method 1: Elapsed time: %.5f sec\n",elapsed);
        printf("   *** %g million operations per second ***\n\n",mops);
    }

    if (1) {
        printf("\nMethod 2 (simple loops - method 2)...\n");
        QTime timer; timer.start();
        method2(M,N,T,Xin,Xout,Kern);
        double elapsed=timer.elapsed()*1.0/1000;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        double mops=M*N*T/elapsed/1e6;
        printf("Method 2: Elapsed time: %.5f sec\n",elapsed);
        printf("   *** %g million operations per second ***\n\n",mops);
    }

    if (1) {
        printf("\nMethod 3 (cblas level 1)...\n");
        QTime timer; timer.start();
        method3(M,N,T,Xin,Xout,Kern);
        double elapsed=timer.elapsed()*1.0/1000;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        double mops=M*N*T/elapsed/1e6;
        printf("Method 3: Elapsed time: %.5f sec\n",elapsed);
        printf("   *** %g million operations per second ***\n\n",mops);
    }

    if (1) {
        printf("\nMethod 4 (cblas level 2)...\n");
        QTime timer; timer.start();
        method4(M,N,T,Xin,Xout,Kern);
        double elapsed=timer.elapsed()*1.0/1000;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        double mops=M*N*T/elapsed/1e6;
        printf("Method 4: Elapsed time: %.5f sec\n",elapsed);
        printf("   *** %g million operations per second ***\n\n",mops);
    }

    if (1) {
        printf("\nMethod 5 (cblas level 2, multiple threads)...\n");
        QTime timer; timer.start();
        method5(M,N,T,Xin,Xout,Kern);
        double elapsed=timer.elapsed()*1.0/1000;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        double mops=M*N*T/elapsed/1e6;
        printf("Method 5: Elapsed time: %.5f sec\n",elapsed);
        printf("   *** %g million operations per second ***\n\n",mops);
    }

    free(Xin);
    free(Xout);
    free(Kern);


    return 0;
}
