#include "imGen.hpp"
/*
 * You can find more info for each function
 * on "imGen.hpp"
*/

#define Iret(i,j) Iret [ (i)*newS + (j) ]
#define I(i,j)    I    [ (i)*N    + (j) ]

inline void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}

int x, y;

int width, height;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;


 float * padarray(float *I,int N,int patch){
   /*Padds the input images with zeros*/
     int newS = N+patch-1;
     float *Iret = (float *)malloc(newS*newS*sizeof(float));

     for(int i=0; i< newS; i++){
         for(int j=0; j< newS;j++){
             if( i < patch/2 || i >= newS - patch/2)
                 Iret[i*newS+j] = 0;
             else if( j < patch/2 || j >= newS - patch/2)
                 Iret[i*newS+j] = 0;
             else
                Iret[i*newS+j] = I[ (i-patch/2)*N + j-patch/2 ]; //I(i-patch/2,j)

         }
     }
     return Iret;
 }

 /*Same functionality with the mirrored padding
  *in the given matlab code on file "non_local_means.m"
  */
 float * padarrayMir(float *I,int N,int patch){

     int newS = N+patch-1;
     float *Iret = (float *)malloc(newS*newS*sizeof(float));

     /*Upper left corner is the origin location*/
     for(int i=0; i< newS; i++){
         for(int j=0; j< newS;j++){
/*Upper left corner*/
            if( i < patch/2 && j < patch/2 )
                Iret(i,j) = I( patch/2-1 - i, patch/2-1 - j );  //Iret(i,j) = I(p/2-1-i,p/2-1-j)
/*Upper middle*/
            else if ( i < patch/2 && j >= patch/2 && j < newS - patch/2 )
                Iret(i,j) = I( patch/2-1 - i, j-patch/2);
/*Upper right corner*/
            else if ( i < patch/2 && j >= newS - patch/2 ){
                int temp = j - N-1 - patch/2;
                Iret(i,j) = I( patch/2 - i - 1 , N-1 - temp - 1 );
            }
/*Middle left*/
            else if ( i >= patch/2 && i <= newS - patch/2 -1 && j < patch/2 )
                Iret(i,j) = I( i - patch/2 , patch/2-1 - j );
/*Middle right*/
            else if ( i >= patch/2 && i <= newS - patch/2-1 && j >= newS - patch/2 )
                Iret(i,j) = I( i - patch/2 , N-1 - patch/2 + newS - j);
/*Down left corner*/
            else if ( i >= newS - patch/2 && j < patch/2 )
                 Iret(i,j) = I(N-1-i+newS-patch/2, patch/2 - j - 1 );
/*Down middle*/
            else if ( i >= newS - patch/2 && j >= patch/2 && j < newS - patch/2 )
                Iret(i,j) = I( N-1 - i + newS - patch/2 , j-patch/2 ); // I( N-1-i, j - patch/2 );
/*Down right corner*/
            else if ( i >= newS - patch/2-1 && j >= newS - patch/2 ){
                int temp = j - N-1 - patch/2;
                Iret(i,j) = I( N-1 - i + newS - patch/2 , N-1 - temp -1  ); // 45;
            }

/*Inside the original matrix*/
            else
                Iret(i,j) = I( i-patch/2, j-patch/2 );

         }
     }

     return Iret;
 }

/*
 * Removes the corners of input image
 * input image dimension npad*npad
 * output image dimension n*n
 */
 float *unpad(float *I, int n,int npad,int patch){
    float *Iun = mymalloc<float>(n*n);
    int ind = 0;

    for (int i = 0; i < npad; i++){
        for(int j=0; j<npad;j++){
            if( i < patch/2 && j < patch/2 )
              { /*Upper left corner*/  }
            else if ( i < patch/2 && j >= patch/2 && j < npad - patch/2 )
              { /*Upper middle*/ }
            else if ( i < patch/2 && j >= npad - patch/2 )
              { /*Upper right corner*/ }
            else if ( i >= patch/2 && i <= npad - patch/2 -1 && j < patch/2 )
              { /*Middle left*/ }
            else if ( i >= patch/2 && i <= npad - patch/2-1 && j >= npad - patch/2 )
              {  /*Middle right*/}
            else if ( i >= npad - patch/2 && j < patch/2 )
              {/*Down left corner*/}
            else if ( i >= npad - patch/2 && j >= patch/2 && j < npad - patch/2 )
               { /*Down middle*/  }
            else if ( i >= npad - patch/2-1 && j >= npad - patch/2 )
            { /*Down right corner*/ }
            else /*Inside the original matrix*/
                Iun[ind++] = I[i*npad+j];
        }
    }

    return Iun;

}

float *unpad2(float *Id, int N, int Npad,int patch){
  float *I = (float*)malloc(N*N*sizeof(float));
  int count = 0;
  int r = patch/2;

  for(int i=0;i<Npad;i++){
    if( i>=(Npad-2*r) || i<0) continue;
    for(int j=0;j<Npad;j++){
      if(j>=(Npad-2*r) || j<0) continue;
      else{
        I[count] = Id[i*Npad+j];
        count++;
      }
    }
  }

  return I;
}

/*Modification of the implementation at https://www.lemoda.net/c/write-png/*/
image * read_png_file(char* file_name)
{
        char header[8];    // 8 is the maximum size that can be checked

        /* open file and test for it being a png */
        FILE *fp = fopen(file_name, "rb");
        if (!fp)
                abort_("[read_png_file] File %s could not be opened for reading", file_name);
        fread(header, 1, 8, fp);
        if (png_sig_cmp((const unsigned char*)header, 0, 8))
                abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


        /* initialize stuff */
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr)
                abort_("[read_png_file] png_create_read_struct failed");

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
                abort_("[read_png_file] png_create_info_struct failed");

        if (setjmp(png_jmpbuf(png_ptr)))
                abort_("[read_png_file] Error during init_io");

        png_init_io(png_ptr, fp);
        png_set_sig_bytes(png_ptr, 8);

        png_read_info(png_ptr, info_ptr);

        width = png_get_image_width(png_ptr, info_ptr);
        height = png_get_image_height(png_ptr, info_ptr);
        color_type = png_get_color_type(png_ptr, info_ptr);
        bit_depth = png_get_bit_depth(png_ptr, info_ptr);

        number_of_passes = png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);


        /* read file */
        if (setjmp(png_jmpbuf(png_ptr)))
                abort_("[read_png_file] Error during read_image");

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (y=0; y<height; y++)
                row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

        png_read_image(png_ptr, row_pointers);

        fclose(fp);

        int ret_image_size = height*width;
        float red, green, blue;
        image * im = (image*)malloc(sizeof(image));

        float *I = (float *)malloc( height * width * sizeof(float) );

        int pixel_size=3;
        if( color_type == PNG_COLOR_TYPE_RGB ) pixel_size = 3;
        if( color_type == PNG_COLOR_TYPE_RGBA) pixel_size = 4;
        if( color_type == PNG_COLOR_TYPE_GRAY) pixel_size = 1;

        for(int i=0;i<height;i++){
          png_byte* row = row_pointers[i];
          for(int j=0;j<width;j++){
            png_byte* ptr = &(row[j*pixel_size]);
            red   = (float) ptr[0]; red   /=255.0;
            green = (float) ptr[1]; green /=255.0;
            blue  = (float) ptr[2]; blue  /=255.0;
            I[i*width+j] =  0.2126 * red + 0.7152 * green + 0.0722 * blue;
          }
        }

        im->I = I;
        im->height = height;
        im->width = width;
        im->color_type = color_type;
        im->bit_depth = bit_depth;

        return im;

}

/*Modification of the implementation at https://www.lemoda.net/c/write-png/*/
int write_png(float *I, size_t height, size_t width, char *path){
  FILE * fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    size_t x, y;
    png_byte ** row_pointers = NULL;
    /* "status" contains the return value of this function. At first
       it is set to a value which means 'failure'. When the routine
       has finished its work, it is set to a value which means
       'success'. */
    int status = -1;


    int pixel_size = 4;
    int depth = 8;

    fp = fopen (path, "wb");
    if (! fp) {
        goto fopen_failed;
    }

    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        goto png_create_write_struct_failed;
    }

    info_ptr = png_create_info_struct (png_ptr);
    if (info_ptr == NULL) {
        goto png_create_info_struct_failed;
    }

    /* Set up error handling. */

    if (setjmp (png_jmpbuf (png_ptr))) {
        goto png_failure;
    }

    /* Set image attributes. */

    png_set_IHDR (png_ptr,
                  info_ptr,
                  width,
                  height,
                  depth,
                  PNG_COLOR_TYPE_RGBA,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);

    /* Initialize rows of PNG. */

    row_pointers = (png_byte**) png_malloc (png_ptr, height * sizeof (png_byte *));
    for (y = 0; y < height; y++) {
        png_byte *row =
            (png_byte*) png_malloc (png_ptr, sizeof (uint8_t) * width * pixel_size);
        //printf("Doing malloc for %d values\n",sizeof (uint8_t) * width * pixel_size);
        row_pointers[y] = row;
        for (x = 0; x < width; x++) {
            uint8_t intensity = (uint8_t) ( I[y*width+x] * 255 );
            *row++ = intensity;
            *row++ = intensity;
            *row++ = intensity;
            *row++ = 255;
        }
    }

    /* Write the image data to "fp". */

    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    /* The routine has successfully written the file, so we set
       "status" to a value which indicates success. */

    status = 0;

    for (y = 0; y < height; y++) {
        png_free (png_ptr, row_pointers[y]);
    }
    png_free (png_ptr, row_pointers);

 png_failure:
 png_create_info_struct_failed:
    png_destroy_write_struct (&png_ptr, &info_ptr);
 png_create_write_struct_failed:
    fclose (fp);
 fopen_failed:
    return status;
}

void writeIm(float *I, const char *fname,int size1, int size2){
  /*Writes image in csv form*/
    FILE *f = fopen(fname,"w");

    for(int i=0;i<size1;i++){
        for(int j=0;j<size2;j++)
            fprintf(f,"%.7f,",I[i*size2+j]);
        fprintf(f,"\n");
    }

    fclose(f);
}

/*
 * Returns a square image (in format float *)
 * read from file 'fname' (Assumes image is stored in csv format)
*/
float *readIm(const char *fname,int size){
  /*Reads image in csv form*/
    FILE *f = fopen(fname,"r");
    float *I = mymalloc<float>(size*size);

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            fscanf(f,"%f,",&I[i*size+j]);
        }

        fscanf(f,"\n");
    }

    fclose(f);

    return I;
}

void printM(float *X, int m,int n){
    for(int i=0;i<m;i++){
        for (int j = 0; j < n; j++)
            printf("%f ",X[i*n+j]);
        std::cout << std::endl;
    }
}










//End Of File
