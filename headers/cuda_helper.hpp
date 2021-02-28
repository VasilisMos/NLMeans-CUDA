#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include "imGen.hpp"
#include "utils.hpp"
#include "parameters.hpp"

#define Iloc(x,y) Iloc[ (x) * n + (y) ]
#define Jnloc(x,y) Jnloc[ (x) * n + (y) ]
#define KERNEL_SIZE ( PATCH * PATCH * sizeof(float) )
#define PAD_IMAGE_SIZE ( Npad * Npad * sizeof(float) )

struct timespec t_start,t_end;

char *path;

/*
 * Error checker after a CUDA API Call used to identify various error
 * source "https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590"
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*CUDA KERNEL ARGUMENTS INITIALISATION*/
int MAX_BLOCKS = B*(6000/(BLOCKS*BLOCKS));
int MAX_THREADS = 1024;

inline dim3 init_blocks(int n, int tiles,int blocks){
  int BLOCKX = n/tiles; int BLOCKY = n/tiles;
  return dim3(min(MAX_BLOCKS,BLOCKX*BLOCKY),1,1);
}
inline dim3 init_threads(int n, int tiles, int blocks){
  int THREADX = blocks;  int THREADY = blocks;
  return dim3(min(MAX_THREADS,THREADX),min(MAX_THREADS,THREADY),1);
}

/*CUDA KERNEL ARGUMENTS INITIALISATION END*/

inline char *init_path(int argc,char *arr){
  char *path = mymalloc<char>(40);
  if(argc == 1)
    strcpy(path,DEFAULT_PATH);
  else
    strcpy(path,arr);

  return path;
}

inline void sendKernel(float *dev, float *host, int patchsize){
  std::cout << "Copying Gaussian kernel to device ..";
  cudaMemcpyToSymbol(dev, host, patchsize * patchsize * sizeof(float) );
  std::cout << "Finished successfully" << std::endl;
}

inline float *get_padded_image(char *fname,int *n,int *npad,int patchsize){
  std::cout << "Starting reading input image..";
  image *im = read_png_file(fname);
  *n = im->height;
  *npad = *n + patchsize -1;

  //Add gaussian noise
  float mean = 0; float std = NOISE_STD;
  addNoise(im->I,mean,std,(*n)*(*n));

  float *temp = mymalloc<float>((*n)*(*n));
  memcpy( temp, im->I, (*n)*(*n)*sizeof(float));

  write_png(temp,*n,*n,"./images/image-Noised-AWGN-GPU-Shared.png");

  float *I = padarrayMir(im->I,*n,patchsize);
  free(im->I); free(im);

  std::cout << "Finished successfully" << std::endl;

  return I;
}

inline void proccess_result(float *Id, int n, int npad, int patchsize){
  std::cout << "Starting writing denoised image to the disc..";
  float *IdFinal = unpad(Id,n,npad,patchsize);

  write_png(IdFinal, n,n,"./images/image-Denoised-GPU-Shared.png");
  std::cout << "Finished successfully" << std::endl;

  writeIm(IdFinal,"./images/txts/image-Denoised-GPU-Shared.txt",n,n);

  free(Id); free(IdFinal);
}


#endif
