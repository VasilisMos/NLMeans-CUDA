#include "imGen.hpp"
#include "utils.hpp"
#include "parameters.hpp"

#define Iloc(x,y) Iloc[x*n+y]
#define Jnloc(x,y) Jnloc[x*n+y]

float *kernel; //Global variable that stores the gaussianKrnl used in computations
char * path;

/*
 * This function gets two pointers in two pixels of an image I
 * and calculates their "neighborhood distance" squared (pixelwise distance of their respective "patchsize" neighborhoods)
 * and returns the exp(-distance/filtersigma) of this distance
 * Note: Each neighborhood is first pixelwise multiplied with a gaussian Kernel
 */
float patchFilt(float *Iloc,float *Jnloc,int n,int patchsize,float filtersigma){
    float dif = 0, *krnl = kernel+(patchsize*patchsize/2); // This kernel now is aligned at the center of the kernel window (and not on the upper right)
    int s = (patchsize/2);

    for(int i=-s;i<=s;i++){
        for(int j=-s; j<=s; j++){
            float wkrnl = krnl[i*patchsize+j];
            dif += wkrnl * wkrnl * (Iloc(i,j) - Jnloc(i,j)) * (Iloc(i,j) - Jnloc(i,j));
        }
    }

    return exp(-dif/filtersigma);
}

/*
 * Sequential Implementaion of non local means algorithm for image denoising
 * Inputs: float *I (Image as a row major float 1D array)
 *         int n    (Size of the image - after the padding has taken place)
 *         int patchsize (defines the patch window size, typical values 3,5,7 (for respective windows [3x3],[5x5],[7x7]))
 *         float filtersigma (used for patchFilt(), for more info see patchFilt())
 * Output: float *Idenoised (Denoised version of input image I with the same dimensions)
 */
float *nlmeansCPU(float *I, int n ,int patchsize, float filtersigma){
    float *Idenoised = mymalloc<float>(n*n);

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

/*
 * Input parser, if 1st command line argument exist is
 * used as path to the png image file to be read
 * else the default image is read
 */
char *init_path(int argc,char *arr){
  char *path = mymalloc<char>(40);
  if(argc == 1)
    strcpy(path,DEFAULT_PATH);
  else
    strcpy(path,arr);

  return path;
}

/*
 * Main function of non local means sequential implementation
 * Pipeline:
 * 1) initialize image path to be read
 * 2) build the gaussian kernel
 * 3) read image
 * 4) add noise to given image
 * 5) mirror pad the noised image
 * 6) perform non local means denoising
 * 7) Remove padded pixels
 * 8) Store noised and denoised images on their respective png files
 * 9) Store execution times and other info
 */
int main(int argc, char *argv[]){
    struct timespec start, end;
    kernel = initKernel(PATCH,patchSigma);
    path = init_path(argc,argv[1]);

    image *im = read_png_file(path);
    int N = im->height,Npad = N+PATCH-1;

    printVersion(0,N,PATCH,R,path);

    //Add gaussian noise
    addNoise(im->I, /*mean*/0, NOISE_STD, N*N);

    //Perform mirror padding
    float *I = padarrayMir(im->I,N,PATCH);

start = tic();
    float *Id = nlmeansCPU(I,Npad,PATCH,filterSigma);
end = toc();

    float *IdFinal = unpad(Id,N,Npad,PATCH);

    /*Store Noised and Denoised Images on disk*/
    write_png(im->I,N,N,"./images/image-Noised-AWGN-seq.png");
    write_png(IdFinal,N,N,"./images/image-Denoised-CPU.png");
    writeIm(IdFinal,"./images/txts/image-Denoised-CPU.txt",N,N);

    free(I); free(Id); free(IdFinal); free(im->I); free(im);

    storeTimes(N,start,end,CPU);

    std::cout << "Main is exiting successfully" << std::endl;
    return 0;
}









//End Of File
