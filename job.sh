#!/bin/bash
#SBATCH --job-name=DeviceQuery
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00


#TEST FOR WINDOW [3x3]
RADIUS=1
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make cuda_global
make cuda_shared

./nlmeans_v1.out "./images/64-house.png"
./nlmeans_v2.out "./images/64-house.png"

./nlmeans_v1.out "./images/128-boat.PNG"
./nlmeans_v2.out "./images/128-boat.PNG"

./nlmeans_v1.out "./images/128-sea.png"
./nlmeans_v2.out "./images/128-sea.png"

./nlmeans_v1.out "./images/256-beach.PNG"
./nlmeans_v2.out "./images/256-beach.PNG"

#TEST FOR WINDOW [5x5]
RADIUS=2
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make cuda_global
make cuda_shared

./nlmeans_v1.out "./images/64-house.png"
./nlmeans_v2.out "./images/64-house.png"

./nlmeans_v1.out "./images/128-boat.PNG"
./nlmeans_v2.out "./images/128-boat.PNG"

./nlmeans_v1.out "./images/128-sea.png"
./nlmeans_v2.out "./images/128-sea.png"

./nlmeans_v1.out "./images/256-beach.PNG"
./nlmeans_v2.out "./images/256-beach.PNG"


#TEST FOR WINDOW [7x7]
RADIUS=3
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make cuda_global
make cuda_shared

./nlmeans_v1.out "./images/64-house.png"
./nlmeans_v2.out "./images/64-house.png"

./nlmeans_v1.out "./images/128-boat.PNG"
./nlmeans_v2.out "./images/128-boat.PNG"

./nlmeans_v1.out "./images/128-sea.png"
./nlmeans_v2.out "./images/128-sea.png"

./nlmeans_v1.out "./images/256-beach.PNG"
./nlmeans_v2.out "./images/256-beach.PNG"
