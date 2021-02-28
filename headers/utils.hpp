#ifndef UTILS_HPP
#define UTILS_HPP

#include <math.h>
#include <time.h>
#include <stdlib.h>

#define PI (2*acos(0.0))
#define CPU 1
#define GPU_GLOBAL 2
#define GPU_SHARED 3

// Returns max (min respectively) of (x,y)
#define max(x,y) ( ((x)>(y)) ? (x) : (y) )
#define min(x,y) ( ((x)<(y)) ? (x) : (y) )

struct timespec t1,t2;

/*
 * Creates a gaussian kernel
 * Similar to a MATLAB command : H = fspecial('gaussian', [n1 n2], s); H = H/max(H(:));
 */
inline void buildGaussKernel(float *krnl,int n1,int n2,float s){

    int h = n1/2;
    for(int i=-h;i<h+1;i++)
        for(int j=-h;j<h+1;j++)
            krnl[ (i+h)*n2+ + (j+h) ] = exp(-(i*i + j*j)/(2*s*s));
}

/*
 * Wrapper used to initialize a square gaussian kernel
 * with height,width equal to 'psize' and sigma value 's'
 */
inline float *initKernel(int psize,float s){
  float *kernel = mymalloc<float>(psize*psize);
  buildGaussKernel(kernel,psize,psize,s);

  return kernel;
}

/*
 * (DEPRECATED)
 * Used on early development to give command line interaction
*/
inline void parseInput(int argc, char *argv[],int *n,int *patch){
    if(argc == 1){
        *n = 64;
        *patch = 5;
    }
    else if(argc == 2){
        *n = atoi(argv[1]);
        *patch = 5;
    }
    else{
        *n = atoi(argv[1]);
        *patch = atoi(argv[2]);
    }

    printf("Image with size |(N,M) = (%d,%d)|patchsize = %d|\n", *n,*n,*patch);
}

/*
 * Prints the time elapsed between two timestamps
 * (Assumes t1 < t2)
*/
void time_elapsed(struct timespec t2, struct timespec t1){
  long seconds = t2.tv_sec - t1.tv_sec;
  long nanoseconds = t2.tv_nsec - t1.tv_nsec;
  if(t2.tv_nsec<t1.tv_nsec){
    seconds--;
    nanoseconds +=1000000000;
  }

  printf("Time elapsed %ld seconds %ld nanoSeconds\n", seconds, nanoseconds);
}

/*
 * Functions that mimic the equivaled MATLAB tic,toc
 * The overhead that the introduce on the time measurement is minimal (few hundred ns)
 * They return the timestamp that is collected using clock_gettime()
*/
inline struct timespec tic() { clock_gettime(CLOCK_MONOTONIC,&t1); return t1; }
inline struct timespec toc() { clock_gettime(CLOCK_MONOTONIC,&t2); time_elapsed(t2,t1); return t2; }

inline void printEvent(const char* s, int DEBUG){
    if(DEBUG) std::cout << s;
}

inline void printEvent(int DEBUG){
    if(DEBUG) std::cout << "Finished successfully" << std::endl;
}

/*
 * Function used to store execution times on version specific .txt file
 * It stores image size (n), seconds, nanoseconds in csv format
*/
inline void storeTimes(int n, struct timespec t1, struct timespec t2, int type){
  FILE *f;
  const char *fname;

  if( type == CPU )               fname = "./results/cputimes.txt";
  else if ( type == GPU_GLOBAL )  fname = "./results/gpu_global_times.txt";
  else if ( type == GPU_SHARED )  fname = "./results/gpu_shared_times.txt";

  f = fopen(fname,"a");

  if ( f == NULL) { printf("Error opening file %s\n", fname ); exit(1);}

  long seconds = t2.tv_sec - t1.tv_sec;
  long nanoseconds = t2.tv_nsec - t1.tv_nsec;
  if(t2.tv_nsec<t1.tv_nsec){
    seconds--;
    nanoseconds +=1000000000;
  }

  fprintf(f, "%d,%d,%ld\n", n, seconds, nanoseconds );

  fclose(f);

}

/*
 * Prints information about the version that is running
 * and the image that is being denoised (size,patchsize,radious=patchsize/2)
*/
inline void printVersion(int vers,int n,int patch,int radious,char * fname){
  const char *version;
  if(vers == 0 ) version = "CPU Version (V0)";
  if(vers == 1 ) version = "GPU Global Version (V1)";
  if(vers == 2 ) version = "GPU Shared Version (V2)";

  printf("\n%s |N=%d|PATCH=%d|R=%d|\n",version,n,patch,radious);
  printf("File: %s\n\n",fname);

}

#endif
