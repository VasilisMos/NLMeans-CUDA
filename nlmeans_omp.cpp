/*
 * CPU OpenMP multithreaded Version
 * Almost identical to nlmeans_seq.cpp
 * with the addition #pragma parallel for inside nlmeansCPU()
 */
#include "imGen.hpp"
#include "utils.hpp"
#include "parameters.hpp"
#include <omp.h>

#define Iloc(x,y) Iloc[x*n+y]
#define Jnloc(x,y) Jnloc[x*n+y]
#define THREADS 4

float *kernel;
char *path;

float patchFilt(float *Iloc,float *Jnloc,int n,int patchsize,float filtersigma){
    float dif = 0, *krnl = kernel+(patchsize*patchsize/2);
    int s = (patchsize/2);

    for(int i=-s;i<=s;i++){
        for(int j=-s; j<=s; j++){
            float wkrnl = krnl[i*patchsize+j];
            dif += wkrnl * wkrnl * (Iloc(i,j) - Jnloc(i,j)) * (Iloc(i,j) - Jnloc(i,j));
        }
    }

    return exp(-dif/filtersigma);
}

float *nlmeansCPU(float *I, int n ,int patchsize, float filtersigma){
    float *Idenoised = mymalloc<float>(n*n);

#pragma omp parallel for shared(Idenoised,I,patchsize,n) num_threads(THREADS)
    for(int xi = patchsize/2; xi < n - patchsize/2; xi++){
        for(int xj = patchsize/2; xj < n - patchsize/2; xj++){
            float z = 0;

            for(int yi=patchsize/2;yi<n-patchsize/2;yi++){
                for(int yj = patchsize/2; yj < n-patchsize/2; yj++){
                    float w = patchFilt(I+(xi*n+xj), I+(yi*n+yj),n, patchsize, filtersigma );
                    z +=w;
                    Idenoised[xi*n+xj] += w*I[yi*n+yj];
                }
            }

            Idenoised[xi*n+xj]/=z;
        }
    }

    return Idenoised;
}

char *init_path(int argc,char *arr){
  char *path = mymalloc<char>(40);
  if(argc == 1)
    strcpy(path,DEFAULT_PATH);
  else
    strcpy(path,arr);

  return path;
}

int main(int argc, char *argv[]){
    struct timespec start, end;
    kernel = initKernel(PATCH,patchSigma);
    path = init_path(argc,argv[1]);

    image *im = read_png_file(path);
    int N = im->height,Npad = N+PATCH-1;

    //Add gaussian noise
    addNoise(im->I, /*mean*/0, NOISE_STD, N*N);

    float *I = padarrayMir(im->I,N,PATCH);

start = tic();

    float *Id = nlmeansCPU(I,Npad,PATCH,filterSigma);
end = toc();

    float *IdFinal = unpad(Id,N,Npad,PATCH);
    write_png(im->I,N,N,"./images/image-Noised-AWGN-omp.png");
    write_png(IdFinal,N,N,"./images/image-Denoised-CPU-omp.png");
    writeIm(IdFinal,"./images/txts/image-Denoised-CPU-omp.txt",N,N);

    free(I); free(Id); free(IdFinal); free(im->I); free(im);

    storeTimes(N,start,end,CPU);

    std::cout << "Main is exiting successfully" << std::endl;
    return 0;
}













//End Of File
