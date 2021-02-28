#include "cuda_helper.hpp"

#define getInd(x,y,n) ( (x) * (n) + (y) ) //Macro to return the (x,y) index of a row major mat. with colSize 'n' into 1D representation (x,y) -> x*n+j
#define ISLEGIT ((threadIdx.x >= R) && (threadIdx.x < (BLOCKS-R)) && (threadIdx.y >= R) && (threadIdx.y < (BLOCKS-R))) //1 for pixels inside the original image, 0 else

__constant__ float gaussianKrnl[PATCH*PATCH];

/* Called From Device - Uses pointers stored on Shared and Constant Memory
 * This function gets two pointers in two pixels of an image I
 * and calculates their "neighborhood distance" squared (pixelwise distance of their respective "patchsize" neighborhoods)
 * and returns the exp(-distance/filtersigma) of this distance
 * Note: Each neighborhood is first pixelwise multiplied with a gaussian Kernel
 */
__device__ float patchFilt(float *Iloc,float *Jnloc,int n,float filtersigma){

    float dif = 0, *krnl = gaussianKrnl+(PATCH*PATCH/2); //This kernel now is aligned at the center of the kernel window (and not on the upper right)

    for(int i=-R;i<=R;i++){
        for(int j=-R; j<=R; j++){
            float wkrnl = krnl[i*PATCH+j];
            dif += wkrnl * wkrnl * (Iloc(i,j) - Jnloc(i,j)) * (Iloc(i,j) - Jnloc(i,j));
        }
    }

    return exp(-dif/filtersigma);
}

__global__ void nlmeans(float *I,float * Idenoised,int n, int TOT_BLOCKS_X, int TOT_BLOCKS_Y,int MAX_BX,int MAX_BY, float filtersigma){
    int width = n+PATCH-1;
    float Idloc,z,w;

    __shared__ float smem[BLOCKS*BLOCKS];
    __shared__ float temp_block[BLOCKS*BLOCKS];

    int blockIdxcpy = blockIdx.x;

    do{
      Idloc = 0, z = 0;
      int blockX = blockIdxcpy / MAX_BX;
      int blockY = blockIdxcpy % MAX_BY;

      int x = blockX*TILES + threadIdx.x - R;
      int y = blockY*TILES + threadIdx.y - R;

      unsigned int index = getInd(x+R, y+R,  width);
      unsigned int bindex = getInd(threadIdx.x, threadIdx.y,  blockDim.y );

      smem[bindex] = I[index];
      __syncthreads();

      //Iterate through all blocks
      for(int iblock=0;iblock<MAX_BX;iblock++){
        for(int jblock=0;jblock<MAX_BY;jblock++){
            //Move to shared memory the (i,j) block

              unsigned int x_temp = getInd(iblock,threadIdx.x , TILES) - R;
              unsigned int y_temp = getInd(jblock,threadIdx.y , TILES) - R;
              index = getInd(x_temp+R,y_temp+R, width);
              temp_block[bindex] = I[index];

              __syncthreads();
              //If point is a legitimate point and not a padded one on the block
              if(ISLEGIT){
                for(int i=R;i<BLOCKS-R;i++){
                  for(int j=R;j<BLOCKS-R;j++){
                    w = patchFilt(smem+(bindex),temp_block+(i*BLOCKS+j),BLOCKS,filtersigma);
                    z += w;
                    Idloc += w*temp_block[i*BLOCKS+j];
                  }
                }
              }
              __syncthreads();
        }
      }

      //If point is a legitimate point and not a padded one on the block
      if(ISLEGIT)  Idenoised[ (x+R)*width + y+R ] = Idloc/z;

      blockIdxcpy += gridDim.x;
    } while( blockIdxcpy <= (n/TILES)*(n/TILES) );

}


int main(int argc, char *argv[]){
    int N, Npad;
    float * d_I, *d_Id;

    path = init_path(argc,argv[1]);
    float *kernel = initKernel(PATCH,patchSigma);

    float *Ipad = get_padded_image(path,&N,&Npad,PATCH);
    float *Id = mymalloc<float>(Npad*Npad);
    printVersion(2,N,PATCH,R,path);

    dim3 blocks = init_blocks(N,TILES,BLOCKS);
    dim3 threads = init_threads(N,TILES,BLOCKS);

    printf("Launching Kernel with BLOCKS:%d (MAX_BLOCKS=%d), THREADS:%d(MAX_THREADS=%d)\n",blocks.x, MAX_BLOCKS, threads.x*threads.y,MAX_THREADS);

    {
      cudaMemcpyToSymbol(gaussianKrnl, kernel, KERNEL_SIZE );
      cudaMalloc( (void **)&d_I, PAD_IMAGE_SIZE );
      cudaMalloc( (void **)&d_Id, PAD_IMAGE_SIZE );
      cudaMemcpy( d_I, Ipad, PAD_IMAGE_SIZE, cudaMemcpyHostToDevice ); free(Ipad);}

t_start = tic();
    nlmeans<<<blocks,threads>>>(d_I,d_Id,N,sqrt(blocks.x),sqrt(blocks.x),N/TILES,N/TILES, filterSigma);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
t_end = toc();

    cudaMemcpy( Id, d_Id, PAD_IMAGE_SIZE, cudaMemcpyDeviceToHost ); cudaFree(d_Id); cudaFree(d_I);

    proccess_result(Id, N, Npad, PATCH);

    std::cout << "Main is exiting successfully" << std::endl;
    return 0;
}







//End Of File
