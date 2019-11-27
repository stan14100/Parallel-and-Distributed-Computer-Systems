#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <game-of-life.h>
#include <mpi.h>

/* set everthing to zero */

void initialize_board (int *board, int N,int M) {

struct timeval startwtime, endwtime; // time counting variables

gettimeofday (&startwtime, NULL);//start timer
  int   i, j;
#pragma omp parallel for private(i,j)
  for (i=0; i<N; i++)
    for (j=0; j<M; j++)
      Board(i,j) = 0;

gettimeofday (&endwtime, NULL); //stop timer
double rearrange_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
 printf("Time to initialize the board : %fs\n", rearrange_time);
}

/* generate random table */

void generate_table (int *board, int N,int M,float threshold) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);//get my rank
  struct timeval startwtime, endwtime;//time counting variables

  gettimeofday (&startwtime, NULL); //start timer

    int   i, j;
    int counter = 0;
    struct drand48_data Buffer;
    double randomvalue;

 #pragma omp parallel private(i,j,Buffer,randomvalue)
  {
    long int seed = time(NULL)*(omp_get_thread_num()+1)*(rank*3+10);  //rank>8 gia na min alliloepikalyptetai me to id toy thread
    srand48_r(seed,&Buffer);
 #pragma omp for reduction(+:counter)
  for (i=0; i<N; i++) {
  	for (j=0; j<M; j++) {
  	 drand48_r(&Buffer, &randomvalue);
     Board(i,j) = (randomvalue / 1.0 )  < threshold;
        counter += Board(i,j);
      }
    }
 }
   gettimeofday (&endwtime, NULL);// stop timer
    double rearrange_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
  				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to generate the table : %fs\n", rearrange_time);
}
