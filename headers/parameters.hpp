#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

//Definition of the standard deviation that we use for our gaussian additive noise
#define NOISE_STD 0.0

//Non local means filtering parameters (same use as their counterparts on nonlocalmeans.m)
#define filterSigma  0.02
#define patchSigma 5.0/3.0

//Tile parameters necessary for V2
#define TILES 16
#define R 2 //Radius of the patch window
#define BLOCKS (TILES+(2*R))

//Patch size (defined as follows)
#define PATCH (2*(R)+1)

//Parameters that controls the amount of blocks are spawned with the formula B*(6000/(BLOCKS*BLOCKS)),
// needs fine tuning for each GPU, locally the optimal value was 2, on HPC the best results were obtained for B>=5
#define B 2

#define DEFAULT_PATH "./images/64-house.png"

using namespace std;

#endif
