#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "utils.h"
#include <pthread.h>
struct morton {
	unsigned long int *morton_codes;
	unsigned int *hash_codes;
	int N;
	int maxlev;
	int NUM_THREADS;
	int id;
};
struct data_rearrangement{
	float *X;
	float *Y;
	unsigned int *permutation_vector;
	int N;
	int NUM_THREADS;
	int id;	
};

#define DIM 3

int main(int argc, char** argv){
int rc;
  // Time counting variables 
  struct timeval startwtime, endwtime;
    
  if (argc != 7) { // Check if the command line arguments are correct 
    printf("Usage: %s N dist pop rep P\n"
	   "where\n"
	   "N    : number of points\n"
	   "dist : distribution code (0-cube, 1-sphere)\n"
	   "pop  : population threshold\n"
	   "rep  : repetitions\n"
	   "L    : maximum tree height.\n"
       "NUM_THREADS: number of threads", argv[0]);
    return (1);
  }

  // Input command line arguments
  int N = atoi(argv[1]); // Number of points
  int dist = atoi(argv[2]); // Distribution identifier 
  int population_threshold = atoi(argv[3]); // populatiton threshold
  int repeat = atoi(argv[4]); // number of independent runs
  int maxlev = atoi(argv[5]); // maximum tree height
  NUM_THREADS= atoi(argv[6]);
  
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  struct morton morton_inputs[NUM_THREADS];
  struct data_rearrangement data_rearrangement_inputs[NUM_THREADS];
   printf("Running for %d particles with maximum height: %d and number of threads: %d\n", N, maxlev,NUM_THREADS);

  float *X = (float *) malloc(N*DIM*sizeof(float));
  float *Y = (float *) malloc(N*DIM*sizeof(float));

  unsigned int *hash_codes = (unsigned int *) malloc(DIM*N*sizeof(unsigned int));
  unsigned long int *morton_codes = (unsigned long int *) malloc(N*sizeof(unsigned long int));
  unsigned long int *sorted_morton_codes = (unsigned long int *) malloc(N*sizeof(unsigned long int));
  unsigned int *permutation_vector = (unsigned int *) malloc(N*sizeof(unsigned int)); 
  unsigned int *index = (unsigned int *) malloc(N*sizeof(unsigned int));
  unsigned int *level_record = (unsigned int *) calloc(N,sizeof(unsigned int)); // record of the leaf of the tree and their level

  // initialize the index
  for(int i=0; i<N; i++){
    index[i] = i;
  }

  /* Generate a 3-dimensional data distribution */
  create_dataset(X, N, dist);

  /* Find the boundaries of the space */
  float max[DIM], min[DIM];
  find_max(max, X, N);
  find_min(min, X, N);

  int nbins = (1 << maxlev); // maximum number of boxes at the leaf level

  // Independent runs
  for(int it = 0; it<repeat; it++){

    gettimeofday (&startwtime, NULL); 
  
    compute_hash_codes(hash_codes, X, N, nbins, min, max, NUM_THREADS); // compute the hash codes

    gettimeofday (&endwtime, NULL);

    double hash_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    
    printf("Time to compute the hash codes            : %fs\n", hash_time);


    gettimeofday (&startwtime, NULL);
    for(int i=0; i<NUM_THREADS; i++){
    morton_inputs[i].morton_codes=morton_codes;
	morton_inputs[i].hash_codes=hash_codes;
	morton_inputs[i].N=N;
	morton_inputs[i].maxlev=maxlev;
	morton_inputs[i].NUM_THREADS=NUM_THREADS;
    morton_inputs[i].id=i;
	rc=pthread_create(&threads[i],&attr, morton_encoding ,(void *) &morton_inputs[i]);
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
    gettimeofday (&endwtime, NULL);


    double morton_encoding_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);


    printf("Time to compute the morton encoding       : %fs\n", morton_encoding_time);


    gettimeofday (&startwtime, NULL); 

    // Truncated msd radix sort
    truncated_radix_sort(morton_codes, sorted_morton_codes, 
			 permutation_vector, 
			 index, level_record, N, 
			 population_threshold, 3*(maxlev-1), 0);

    gettimeofday (&endwtime, NULL);

    double sort_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    printf("Time for the truncated radix sort         : %fs\n", sort_time);

    gettimeofday (&startwtime, NULL); 
    for(int i=0; i<NUM_THREADS; i++){
    	data_rearrangement_inputs[i].X=X;
		data_rearrangement_inputs[i].Y=Y;
		data_rearrangement_inputs[i].permutation_vector=permutation_vector;
		data_rearrangement_inputs[i].N=N;
		data_rearrangement_inputs[i].NUM_THREADS=NUM_THREADS;
		data_rearrangement_inputs[i].id=i;
		rc=pthread_create(&threads[i],&attr, data_rearrangement,(void *) &data_rearrangement_inputs[i]); // Data rearrangement
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
    gettimeofday (&endwtime, NULL);


    double rearrange_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    

    printf("Time to rearrange the particles in memory : %fs\n", rearrange_time);

    /* The following code is for verification */ 
    // Check if every point is assigned to one leaf of the tree
    int pass = check_index(permutation_vector, N); 

    if(pass){
      printf("Index test PASS\n");
    }
    else{
      printf("Index test FAIL\n");
    }

    // Check is all particles that are in the same box have the same encoding. 
    pass = check_codes(Y, sorted_morton_codes, 
		       level_record, N, maxlev);

    if(pass){
      printf("Encoding test PASS\n");
    }
    else{
      printf("Encoding test FAIL\n");
    }

  }

  /* clear memory */
  free(X);
  free(Y);
  free(hash_codes);
  free(morton_codes);
  free(sorted_morton_codes);
  free(permutation_vector);
  free(index);
  free(level_record);
}





