#include "cuda_helper.hpp"

#define Iloc(x,y) Iloc[x*n+y]
#define Jnloc(x,y) Jnloc[x*n+y]

#define K 2

__constant__ float gkrnl[PATCH*PATCH];

__device__ float patchFilt(float *Iloc,float *Jnloc,int n,float filtersigma,float *a){

   float dif = 0;

   for(int i=-R;i<=R;i++){
       for(int j=-R; j<=R; j++){
           float wkrnl = a[i*PATCH+j + (PATCH*PATCH/2)];
           dif += wkrnl * wkrnl * (Iloc(i,j) - Jnloc(i,j)) * (Iloc(i,j) - Jnloc(i,j));
       }
   }

   return exp(-dif/filtersigma);
}

/*This kernel launches with row blocks and column threads*/
__global__ void nlmeans(float *I,
                       float *Idenoised,
                       int n,
                       int patchsize,
                       float filtersigma){

   int xj = threadIdx.x;
   int xi = K*blockIdx.x ;

   __shared__ float kernel_shared[PATCH*PATCH];
   if(threadIdx.x <PATCH*PATCH) kernel_shared[threadIdx.x] = gkrnl[threadIdx.x];

   __syncthreads();

   for(int i=0;i<K;i++,xi++){
       float Idloc = 0,z = 0;

       if( (xi >= R)  &&   (xi < n-R)   &&   (xj >= R)  &&  (xj < n-R) ){

         for(int yi=R;yi<n-R;yi++){
             for(int yj = R; yj < n-R; yj++){
                 float w = patchFilt(I+(xi*n+xj), I+(yi*n+yj),n, filtersigma,kernel_shared);
                 z +=w;
                 Idloc += w*I[yi*n+yj];
             }
         }
         Idenoised[xi*n+xj] = Idloc/z;
     }
   }
}

int main(int argc, char *argv[]){
   float * d_I, *d_Id,*gaussianKrnl;

   path = init_path(argc,argv[1]);
   float *kernel = initKernel(PATCH,patchSigma);
   image *im = read_png_file(path); int N = im->height, Npad = N+PATCH-1;

   //Add gaussian noise
   float mean = 0; float std = NOISE_STD;
   addNoise(im->I,mean,std,N*N);

   float *I = padarrayMir(im->I,N,PATCH);
   float *Id = mymalloc<float>(Npad*Npad);
   printVersion(2,N,PATCH,R,path);

   printf("Launching Kernel with BLOCKS:%d (MAX_BLOCKS=%d), THREADS:%d(MAX_THREADS=%d)\n",Npad, MAX_BLOCKS, Npad,MAX_THREADS);

   cudaMalloc( (void **)&d_I, sizeof(float) * Npad * Npad );
   cudaMalloc( (void **)&d_Id, sizeof(float) * Npad * Npad );
   cudaMemcpy( d_I, I, sizeof(float) * Npad * Npad, cudaMemcpyHostToDevice );
   cudaMemcpyToSymbol(gkrnl, kernel, KERNEL_SIZE );

struct timespec start = tic();

   nlmeans<<<Npad/K,Npad>>>(d_I,d_Id,Npad, PATCH , filterSigma);

   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );

struct timespec end = toc();

   cudaMemcpy( Id, d_Id, sizeof(float) * Npad * Npad, cudaMemcpyDeviceToHost );

   float *IdFinal = unpad(Id,N,Npad,PATCH);
   write_png(im->I,N,N,"./images/image-Noised-AWGN-GPU-Shared.png");
   write_png(IdFinal,N,N,"./images/image-Denoised-GPU-Shared.png");


   cudaFree(d_Id); cudaFree(d_I);
   free(I); free(Id); free(IdFinal); free(im->I); free(im);

   storeTimes(N,start,end,GPU_SHARED);

   std::cout << "Main is exiting successfully" << std::endl;
   return 0;
}




//End Of File
