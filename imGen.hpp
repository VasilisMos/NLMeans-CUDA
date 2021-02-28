#ifndef IMGEN_HPP
#define IMGEN_HPP

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <png.h>

#define max(x,y) ( ((x)>(y)) ? (x) : (y) )
#define min(x,y) ( ((x)<(y)) ? (x) : (y) )

/*
 * Image Struct, contains:
 * float *I (Array that stores pixel values (in range [0,1]))
 * height, width (dimensions of image)
 * color_type (parameter needed for png manipulations, specifies the different color_type png format supports (RGB,RGBA,etc))
 * bit_depth (parameter needed for png manipulations, specifies the bit_depth of the png output or input image)
 */
typedef struct image{
  float * I; //intensity
  int height;
  int width;
  int color_type;
  int bit_depth;
} image;

/*
 * Templated version of malloc for any primitive data type
 * Reserves memory equal to 'size' times the sizeof the given data type
 * and returns a pointer at the 1st element
*/
template <typename T>
inline T* mymalloc(int size) { return (T*)malloc(size*sizeof(T));}

/*
 *Return a random array with length equal to 'size'
 *Each value of the array is within the range of [0,1)
*/
template <typename T>
inline T *randM(int size){
    srand(time(NULL));
    T *ret = mymalloc<T>(size);
    for(int i=0;i<size;i++) { ret[i] = (T)((rand()%100)/100.0);}
    return ret;
}

/*
 * Return a square image (as a row major 1D float array) with side
 * length equal to 'size'
 * Each value of the array is within the range of [0,1)
*/
inline float *randIMG(int size){ return randM<float>(size*size); }

/*
 * Adds AWGN (Additive White Gaussian Noise) on a given float array
 * with mean and standard deviation given as inputs.
 * Since 'data' input array is planned to be an image
 * the noised data are bounded to [0,1]
*/
inline void addNoise(float *data, float mean, float std,int n){
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, std);

    if (std == 0) return;

    // Add Gaussian noise
    for (int i=0;i<n;i++){
      data[i] = data[i] + dist(generator);
      data[i] = max(data[i],0);
      data[i] = min(data[i],1);
    }

}

float * padarray(float *I,int N,int patch);
float * padarrayMir(float *I,int N,int patch);
float * unpad(float *I, int n,int npad,int patch);
float *unpad2(float *Id, int N, int Npad,int patch);

/*
 * Functions designed to parse (from .txt)
 * and write back (from float arr to .txt) images,
 */
void writeIm(float *I, const char *fname,int size1,int size2);
float *readIm(const char *fname,int size);

/*
 * Functions designed to parse and write back png images,
 * based on the initial implementation at https://www.lemoda.net/c/write-png/
 * Modified to meet this app needs and to cover more png colour Types
 */
image * read_png_file(char* file_name);
int write_png(float *I, size_t height, size_t width, char *path);

/*
 * Prints a 2D matrix X with rowSize m and columnSize n
 */
void printM(float *X, int m,int n);


#endif
