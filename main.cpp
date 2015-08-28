//#include <QCoreApplication>
#include <stdio.h>
#include "cblas.h"
#include "omp.h"
#include <time.h>
#include <stdlib.h>

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
//void method5(int M,int N,int T,double *Xin,double *Xout,double *K2) {

//    for (int n=0; n+T-1<N-T; n+=T) {
//        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,T,2*T-1,1.0,&Xin[M*n],M,K2,(2*T-1),0,&Xout[M*n],M);
//    }
//}

void method5(int M,int N,int T,double *Xin,double *Xout,double *Kern) {

    omp_set_num_threads(40);
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

    int M=512;
    int N=1200000;
    int T=100;

    double *Xin=(double *)malloc(sizeof(double)*M*N);
    double *Xout=(double *)malloc(sizeof(double)*M*N);
    double *Kern=(double *)malloc(sizeof(double)*T);

    printf("Preparing...\n");
    for (int ii=0; ii<M*N; ii++) {
        Xin[ii]=ii%183;
    }
    for (int t=0; t<T; t++) {
        Kern[t]=rand()*1.0/rand();
    }

    if (0) {
        printf("\nMethod 1...");
        clock_t T1=clock();
        method1(M,N,T,Xin,Xout,Kern);
        clock_t T2=clock();
        double elapsed=((double)(T2-T1))/CLOCKS_PER_SEC;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        printf("Method 1: Elapsed time: %.5f sec\n",elapsed);
    }

    if (0) {
        printf("\nMethod 2...");
        clock_t T1=clock();
        method2(M,N,T,Xin,Xout,Kern);
        clock_t T2=clock();
        double elapsed=((double)(T2-T1))/CLOCKS_PER_SEC;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        printf("Method 2: Elapsed time: %.5f sec\n",elapsed);
    }

    if (0) {
        printf("\nMethod 3...");
        clock_t T1=clock();
        method3(M,N,T,Xin,Xout,Kern);
        clock_t T2=clock();
        double elapsed=((double)(T2-T1))/CLOCKS_PER_SEC;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        printf("Method 3: Elapsed time: %.5f sec\n",elapsed);
    }

    if (0) {
        printf("\nMethod 4...");
        clock_t T1=clock();
        method4(M,N,T,Xin,Xout,Kern);
        clock_t T2=clock();
        double elapsed=((double)(T2-T1))/CLOCKS_PER_SEC;
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        printf("Method 4: Elapsed time: %.5f sec\n",elapsed);
    }

    if (1) {
        printf("\nMethod 5...");
        struct timespec T1,T2;
        clock_gettime(CLOCK_REALTIME, &T1);
        method5(M,N,T,Xin,Xout,Kern);
        clock_gettime(CLOCK_REALTIME, &T2);
        double elapsed=((double)((T2.tv_sec+T2.tv_nsec/1e9)-(T1.tv_sec+T1.tv_nsec/1e9)));
        printf("%g,%g,%g\n",Xout[1],Xout[1000],Xout[100000]);
        printf("Method 5: Elapsed time: %g\n",elapsed);
    }

    free(Xin);
    free(Xout);
    free(Kern);


    return 0;

    //return a.exec();
}
