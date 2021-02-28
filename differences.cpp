/* RUN with >make validation
 * This file is used to validate the agreement on results between 2 versions
 * of non local means implementation.
 *
 * Someone can check the equality of 2 versions by specifying the corresponding .txts below on paths 1 and 2 and
 * declaring the image row pixel size N (e.g 64,128,256)
 * for sequential version path: "./images/txts/image-Denoised-CPU.txt"
 * for cuda global path: "./images/txts/image-Denoised-GPU-Global.txt"
 * for cuda shared path: "./images/txts/image-Denoised-GPU-Shared.txt"
 * This validation checks for errors larger than 9*10^-5, because below that are not detectable after the png quantization
 * NOTE!!!: Before running this validation you should run the respective versions with the same image and the NOISE_STD set to 0
 * to get meaningful results (without noise still filtering occurs - no noise for reproducibility )
 */

#include "imGen.hpp"

#define path1 "./images/txts/image-Denoised-CPU.txt"
#define path2 "./images/txts/image-Denoised-GPU-Shared.txt"
#define N 64

int main(){
  int count = 0;

  float *I1 = readIm(path1,N);
  float *I2 = readIm(path2,N);

  float *difs = (float*)malloc(N*N*sizeof(float));

  for(int i=0;i<N*N;i++) difs[i] = I1[i] - I2[i];

  //Checking for errors at least 9*10^-5, below that errors are not detectable after the png quantization
  for(int i=0;i<N*N;i++) if(abs(difs[i]) >= 0.0001) count++;

  if(count ==0) printf("Validation SUCCESSFUL,");
  else printf("Validation FAILED,");

  printf(" Error in: %d pixels out of %d\n",count, N*N);
}



//End of File
