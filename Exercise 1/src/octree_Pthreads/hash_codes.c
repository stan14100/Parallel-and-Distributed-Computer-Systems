#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include <pthread.h> 

#define DIM 3
struct quantize_str{
   int id;
   float qstep;
   float *min;
   int N;
   unsigned int *hash_codes;
   float *X;
   int NUM_THREADS;
};

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}


/* Function that does the quantization */
void *quantize(void * st){
   struct quantize_str *my_data;
   my_data=(struct quantize_str *) st;
   int a=(int) (my_data->N/my_data->NUM_THREADS);
    int start=(my_data->id)*(a),stop=(my_data->id+1)*(a);
    if(my_data->id==(my_data->NUM_THREADS-1)) stop=my_data->N;
  for(int i=start; i<stop; i++){
    for(int j=0; j<DIM; j++){
      my_data->hash_codes[i*DIM + j] = compute_code(my_data->X[i*DIM + j], my_data->min[j],my_data->qstep); 
    }
  }
pthread_exit(NULL);
}

float max_range(float *x){

  float max = -FLT_MAX;
  for(int i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, float *max, int NUM_THREADS ){
	int rc;
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  struct quantize_str p[NUM_THREADS];
  float range[DIM];
  float qstep;
  

 for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
  }
  
  qstep = max_range(range) / nbins; // The quantization step 
  for(int i=0; i<NUM_THREADS; i++){
   p[i].id=i;
   p[i].qstep=qstep;
   p[i].min=min;
   p[i].N=N;
   p[i].hash_codes=codes;
   p[i].X=X;
   p[i].NUM_THREADS=NUM_THREADS;
   rc=pthread_create(&threads[i],&attr,quantize,(void *) &p[i]);
   if (rc)
   	            {
   	                printf("Error: pthread_create returned code %d\n", rc);
   	                return;
   	}
  }
  for(int i=0;i<NUM_THREADS;i++) {
  rc=pthread_join(threads[i],NULL);
  if (rc) {
                            printf("Error: pthread_join returned code %d\n", rc);
                            return;
            }
  }
  pthread_attr_destroy(&attr);
}



