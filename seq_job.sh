#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --time=00:07:00

#TEST FOR WINDOW [3x3]
RADIUS=1
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make sequential

./nlmeans_seq.out "./images/64-house.png"
./nlmeans_seq.out "./images/128-sea.png"
./nlmeans_seq.out "./images/256-beach.PNG"

#TEST FOR WINDOW [5x5]
RADIUS=2
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make sequential

./nlmeans_seq.out "./images/64-house.png"
./nlmeans_seq.out "./images/128-sea.png"
./nlmeans_seq.out "./images/256-beach.PNG"


#TEST FOR WINDOW [7x7]
RADIUS=3
bash setParams.sh ${RADIUS}

module purge
module load gcc/7.3.0 cuda/10.0.130 libpng
make sequential

./nlmeans_seq.out "./images/64-house.png"
./nlmeans_seq.out "./images/128-sea.png"
./nlmeans_seq.out "./images/256-beach.PNG"
