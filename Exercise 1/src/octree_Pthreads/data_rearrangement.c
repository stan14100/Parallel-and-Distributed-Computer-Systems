#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include<pthread.h>

struct data_rearrangement{
	float *X;
	float *Y;
	unsigned int *permutation_vector;
	int N;
	int NUM_THREADS;
	int id;	
};
extern struct data_rearrangement data_rearrangement_inputs;

#define DIM 3


void *data_rearrangement(void *st){
	struct data_rearrangement *my_data;
	   my_data=(struct data_rearrangement *) st;
	   int a=(int) (my_data->N/my_data->NUM_THREADS);
	    int start=(my_data->id)*(a),stop=(my_data->id+1)*(a);
	    if(my_data->id==(my_data->NUM_THREADS-1)) stop=my_data->N;
  for(int i=start; i<stop; i++){
    memcpy(&my_data->Y[i*DIM], &my_data->X[my_data->permutation_vector[i]*DIM], DIM*sizeof(float));
  }
  
pthread_exit(NULL);
}
