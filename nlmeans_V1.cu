#include "cuda_helper.hpp"

#define Iloc(x,y) Iloc[x*n+y]
#define Jnloc(x,y) Jnloc[x*n+y]

/* Called From Device - Uses pointers stored on Global Memory only
 * This function gets two pointers in two pixels of an image I
 * and calculates their "neighborhood distance" squared (pixelwise distance of their respective "patchsize" neighborhoods)
 * and returns the exp(-distance/filtersigma) of this distance
 * Note: Each neighborhood is first pixelwise multiplied with a gaussian Kernel
 */
__device__ float patchFilt(float *Iloc,float *Jnloc,int n,float filtersigma,float *gaussianKrnl){
    int offs = PATCH*PATCH/2;
    float dif = 0, *krnl = gaussianKrnl+(offs); //This kernel now is aligned at the center of the kernel window (and not on the upper right)

    for(int i=-R;i<=R;i++){
        for(int j=-R; j<=R; j++){
            float wkrnl = krnl[i*PATCH+j];
            dif += wkrnl * wkrnl * (Iloc(i,j) - Jnloc(i,j)) * (Iloc(i,j) - Jnloc(i,j));
        }
    }

    return exp(-dif/filtersigma);
}

/*
 * CUDA Kernel that performs the non local means denoisation (Uses Global memory only)
 * Inputs-Outputs: float *I (Image as a row major float 1D array) - Global Memory
 *         float *I (Output Denoised image stored in row major 1D format) -Global Memory
 *         int n    (Size of the image - after the padding has taken place)
 *         int patchsize (defines the patch window size, typical values 3,5,7 (for respective windows [3x3],[5x5],[7x7]))
 *         float filtersigma (used for patchFilt(), for more info see patchFilt())
 *         float *gaussianKrnl (pointer to the gaussian kernel that is multiplied with each neighborhood) - Global Memory
 *
 */
__global__ void nlmeans(float *I,
                        float *Idenoised,
                        int n,
                        int patchsize,
                        float filtersigma,
                        float *gaussianKrnl){

    /*Each pixel is mapped on a block-thread (Valid assumption for Images with dimensions as such on the assignment (N=64~256))*/
    int xi = threadIdx.x;
    int xj = blockIdx.x ;

    float Idloc = 0,z = 0;

    //If the threads refers to a pixel inside the original image (not on the padded area)
    //if( (xi >= ph)  &&   (xi < n-ph)   &&   (xj >= ph)  &&  (xj < n-ph) ){
    if( (xi >= R)  &&   (xi < n-R)   &&   (xj >= R)  &&  (xj < n-R) ){

      //Calculate distances of a specific pixel with every other pixel on the (original) image
      for(int yi=R;yi<n-R;yi++){
          for(int yj = R; yj < n-R; yj++){
              float w = patchFilt(I+(xi*n+xj), I+(yi*n+yj),n, filtersigma, gaussianKrnl );
              z +=w;
              Idloc += w*I[yi*n+yj];
          }
      }

      Idenoised[xi*n+xj] = Idloc/z;
    }
}

int main(int argc, char *argv[]){
    float * d_I, *d_Id,*gaussianKrnl;

    path = init_path(argc,argv[1]);

    std::cout << "Starting building Gaussian kernel ..";
    float *kernel = mymalloc<float>(PATCH*PATCH);
    buildGaussKernel(kernel,PATCH,PATCH,patchSigma);
    std::cout << "Finished successfully" << std::endl;

    std::cout << "Starting reading input image..";
    image *im = read_png_file(path); int N = im->height, Npad = N+PATCH-1;

    //Add gaussian noise
    float mean = 0; float std = NOISE_STD;
    addNoise(im->I,mean,std,N*N);

    float *I = padarrayMir(im->I,N,PATCH);
    std::cout << "Finished successfully" << std::endl;
    float *Id = mymalloc<float>(Npad*Npad);

    printVersion(1,N,PATCH,R,path);
    std::cout << "Starting running CUDA nlmeans algorithm..";

    printf("Launching Kernel with BLOCKS:%d (MAX_BLOCKS=%d), THREADS:%d(MAX_THREADS=%d)\n",Npad, MAX_BLOCKS, Npad,MAX_THREADS);

    {
      cudaMalloc( (void **)&gaussianKrnl, sizeof(float) * PATCH * PATCH );
      cudaMemcpy( gaussianKrnl, kernel, sizeof(float) * PATCH * PATCH, cudaMemcpyHostToDevice );

      cudaMalloc( (void **)&d_I, sizeof(float) * Npad * Npad );
      cudaMalloc( (void **)&d_Id, sizeof(float) * Npad * Npad );
      cudaMemcpy( d_I, I, sizeof(float) * Npad * Npad, cudaMemcpyHostToDevice );}

struct timespec start = tic();
    nlmeans<<<Npad,Npad>>>(d_I,d_Id,Npad, PATCH , filterSigma, gaussianKrnl);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
struct timespec end = toc();


    cudaMemcpy( Id, d_Id, sizeof(float) * Npad * Npad, cudaMemcpyDeviceToHost );
    std::cout << "Finished successfully" << std::endl;

    std::cout << "Starting writing denoised image to the disc..";
    float *IdFinal = unpad(Id,N,Npad,PATCH);
    write_png(im->I,N,N,"./images/image-Noised-AWGN-GPU-Global.png");
    write_png(IdFinal,N,N,"./images/image-Denoised-GPU-Global.png");
    writeIm(IdFinal,"./images/txts/image-Denoised-GPU-Global.txt",N,N);
    std::cout << "Finished successfully" << std::endl;

    cudaFree(d_Id); cudaFree(d_I); cudaFree(gaussianKrnl);
    free(I); free(Id); free(IdFinal);
    free(im->I); free(im);

    storeTimes(N,start,end,GPU_GLOBAL);

    std::cout << "Main is exiting successfully" << std::endl;
    return 0;
}
