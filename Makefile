CC=g++
FLAGS=-O3
NVCC=nvcc
ARCH=-arch=sm_50

copy_headers:
		cp ./headers/* .

sequential: copy_headers
	$(CC) $(FLAGS) ./nlmeans_seq.cpp ./imGen.cpp -o nlmeans_seq.out -lpng -w && rm utils.hpp parameters.hpp cuda_helper.hpp && clear

omp: copy_headers
	$(CC) $(FLAGS) ./nlmeans_omp.cpp ./imGen.cpp -o nlmeans_omp.out -lpng -fopenmp -w && rm utils.hpp parameters.hpp cuda_helper.hpp && clear

cuda_global: copy_headers
	$(NVCC) $(ARCH) ./nlmeans_V1.cu ./imGen.cpp -o nlmeans_v1.out -lpng -w && rm utils.hpp parameters.hpp cuda_helper.hpp && clear

cuda_shared: copy_headers
	$(NVCC) $(ARCH) ./nlmeans_V2.cu ./imGen.cpp -o nlmeans_v2.out -lpng -w && rm utils.hpp parameters.hpp cuda_helper.hpp && clear

cuda_shared_new: copy_headers
	$(NVCC) $(ARCH) ./nlmeans_V2new.cu ./imGen.cpp -o nlmeans_v2.out -lpng -w && rm utils.hpp parameters.hpp cuda_helper.hpp && clear

validation: copy_headers
	$(CC) ./differences.cpp ./imGen.cpp -o ./validate.out -lpng && rm utils.hpp parameters.hpp cuda_helper.hpp && clear
	./validate.out && rm ./validate.out

test_seq: sequential
	./nlmeans_seq.out && rm ./nlmeans_seq.out

test_omp: omp
	./nlmeans_omp.out && rm ./nlmeans_omp.out

test_cuda_global: cuda_global
	./nlmeans_v1.out && rm ./nlmeans_v1.out

test_cuda_shared: cuda_shared
	./nlmeans_v2.out && rm ./nlmeans_v2.out

test_cuda_shared_new: cuda_shared_new
	./nlmeans_v2.out && rm ./nlmeans_v2.out

clean:
	rm -f *.out ./images/*Denoised* ./images/*AWGN* ./images/txts/*

purge: clean
	rm -f ./results/*
