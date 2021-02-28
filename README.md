# NLMeans-CUDA
This project implements the well-known Non-local means algorithm that is used for Image Denoising, both in a CPU version (for comparison purposes) and (most interestingly) in a CUDA version that takes advantage of the higher computing capabilities of a GPGPU for this type of problems compared to a conventional multicore CPU. The project was developed for the purposes of the course "Parallel and Distributed Systems", ECE AUTh.

Below is a presentation of each version in more detail (Further explanations on how to execute each version and how code works are given on the report.pdf):

`Sequential Version`: Implementation of Non-local means on CPU

`Multithreaded Version`: Parallel Implementation of Non-local means on CPU using multiple cores with OpenMP library (default thread count is set to 4)

`V1 Version`: CUDA Implementation of Non-local means using only GPU's global memory during computations

`V2 Version`: CUDA implementation of Non-local means that stores Gaussian Kernel on constant memory and also takes advantage of shared memory that GPU provides

## Compilation - Run

To run the below code versions successfully open a terminal in the main folder of the project and place the desired `.png` image on the subdirectory `./images/`. The `libpng` C library should be installed on the system, for more information visit the official site http://www.libpng.org/pub/png/libpng.html. (Note: The output images are in grayscale form and are placed on the `images` dir). The project development was performed on a Ubuntu 18.04 system, using nvcc --version=V9.1.85. To someone who faces cross platform compatibility issues (or issues related to libpng), HPC AUTh infastructure can be used (https://hpc.it.auth.gr/).

Run the sequential version with

    make sequential
    ./nlmeans_seq.out "./images/someImage.png"

 Similarly run the cuda versions with

    make cuda_global
    ./nlmeans_v1.out "./images/someImage.png"
    make cuda_shared
    ./nlmeans_v2.out "./images/someImage.png"

To run each version with the default inputs (E-learning given image with N = 64 and patchsize = 5) simply run

    make test_seq
    make test_cuda_global
    make test_cuda_shared


To run on HPC AUTh infastructure simply copy the whole project folder there and submit:

    sbatch job.sh
    sbatch seq_job.sh



## Performance
Below are some results of nonlocalmeans() execution time for each version.

##### Local Test (Laptop Intel i7-4720HQ, Nvidia GeForce GTX 960M  - Compute Capability 5.0)

Patch size [`3x3`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 400 ms  | 6370 ms| 107031 ms |
| CPU - OpenMP | 122 ms | 2338 ms | 41740 ms  |
| Cuda Global | 6.8 ms | 58 ms | 730 ms |
| Cuda Shared | 4 ms | 50 ms | 724 ms |

Patch size [`5x5`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 715 ms | 13340 ms | 195302 ms |
| CPU - OpenMP | 297 ms | 5720 ms | 77219 ms |
| Cuda Global | 78 ms | 1142 ms | 17631 ms |
| Cuda Shared | 15 ms | 152 ms | 2464 ms |

Patch size [`7x7`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 1190 ms | 20812 ms | 306203 ms
| CPU - OpenMP | 413 ms | 8108 ms | 105540 ms
| Cuda Global | 142 ms | 2215 ms | Kernel timed Out (>30 sec) |
| Cuda Shared | 21 ms | 251 ms | 3610 ms |

##### HPC Test (Nvidia Tesla P100 - Compute Capability 6.0)

Patch size [`3x3`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 584 ms | 8986 ms | 204597 ms |
| CPU - OpenMP | 234 ms | 2607 ms | 56102 ms |
| Cuda Global | 9.4 ms | 51 ms | 522 ms |
| Cuda Shared | 14.4 ms | 82 ms | 488 ms |

Patch size [`5x5`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 990 ms | 15884 ms | 253229 ms |
| CPU - OpenMP | 458 ms | 4460 ms | 67879 ms |
| Cuda Global | 50 ms | 465 ms | 5812 ms |
| Cuda Shared | 35 ms | 207 ms | 912 ms |

Patch size [`7x7`]

| N | 64x64 pixels | 128x128 pixels | 256x256 pixels |
| --- | ----------- | ------------- | ------------- |
| Sequential | 1483 ms | 23597 ms | 283025 ms |
| CPU - OpenMP | 592 ms | 6345 ms | 100118 ms |
| Cuda Global | 84 ms | 659 ms | 11075 ms |
| Cuda Shared | 60 ms | 274 ms | 1418 ms |

## Correctness - Validation
The correctness of the output image that each version produces can be checked on the site <a id="1/">https://online-image-comparison.com/</a> by uploading the image that each version produces (If 2 images are the same the result should not show any red signs - Insert a small 'Fuzz' value to detect even small differences). To verify the equality of two images with C simply run:

    make validation

To decide which 2 versions to validate you have to modify just the 2 filenames at differences.cpp (More detailed info on the file)
Note:You should first run the 2 versions you are about to validate, with 0.0 NOISE_STD (parameters.hpp). All the above images produced (N size: {64,128,256} and patch size {3,5,7}) were checked with the second option which (obviously) is more precise.

### References
<a id="1">[1]</a>
Buades, Antoni (20–25 June 2005).
A non-local algorithm for image denoising.
Computer Vision and Pattern Recognition, 2005
