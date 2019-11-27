#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <omp.h>

#define DIM 3

extern NUM_THREADS;
void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){
#pragma omp parallel for num_threads(NUM_THREADS) //mporoume na to kanoume kai me auton ton tropo
  for(int i=0; i<N; i++){
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
