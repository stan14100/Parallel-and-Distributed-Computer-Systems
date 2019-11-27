#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include<omp.h>
#include<mpi.h>
#include <game-of-life.h>
struct timeval startwtime, endwtime;
/* set everthing to zero */

void initialize_board (int *board, int N, int M) {
  int   i, j;
  gettimeofday (&startwtime, NULL);
#pragma omp parallel for private(j)
  for (i=0; i<N; i++){
    for (j=0; j<M; j++) 
      Board(i,j) = 0;
  
  }
  gettimeofday (&endwtime, NULL);

        double initialize_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
    				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
        
        printf("Time to compute the initialize codes            : %fs\n", initialize_time);
}

/* generate random table */
  
  
  void generate_table (int *board, int N, int M ,float threshold) {
	  int SelfTID;
	   MPI_Comm_rank(MPI_COMM_WORLD,&SelfTID);//get my rank
	   gettimeofday (&startwtime, NULL);
	  int   i, j;
	     int counter = 0;
	     struct drand48_data Buffer;
	     double randomvalue;

	  #pragma omp parallel private(i,j,Buffer,randomvalue)
	   {
	     long int seed = time(NULL)*(omp_get_thread_num()+1)*(SelfTID*3+10);  //8etw to seed na eksartatai apo to thread id gia na mhn uparxei epikalupsh twn random number sta threads
	     srand48_r(seed,&Buffer);                                             //kai epishs logw twn tasks (exw tasks*8threads) gia na mhn exw epikalupsh bazw kapoies prakseis me tuxaious ari8mous gia na 
	  #pragma omp for reduction(+:counter)					 // thn apofugw
	   for (i=0; i<N; i++) {
	         for (j=0; j<M; j++) {
	          drand48_r(&Buffer, &randomvalue);
	      Board(i,j) = (randomvalue / 1.0 )  < threshold;
	         counter += Board(i,j);
	       }
	     }
	  }

  gettimeofday (&endwtime, NULL);

      double generate_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
  				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
      
      printf("Time to compute the generate codes            : %fs\n", generate_time);
}
