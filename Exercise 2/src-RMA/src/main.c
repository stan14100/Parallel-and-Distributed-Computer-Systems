/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include<omp.h>
#include<mpi.h>
#include <game-of-life.h>

int main (int argc, char *argv[]) {
  int   *board, *newboard, i;

  if (argc != 6) { // Check if the command line arguments are correct 
    printf("Usage: %s N M thres disp\n"
	   "where\n"
	   "  N     : size of rows \n"
       "  M     : size of columns\n"
	   "  thres : propability of alive cell\n"
           "  t     : number of generations\n"
	   "  disp  : {1: display output, 0: hide output}\n"
           , argv[0]);
    return (1);
  }
  omp_set_num_threads(8);
  
  // Input command line arguments
  int N = atoi(argv[1]);        // Array size
  int M = atoi(argv[2]);
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations 
  int disp = atoi(argv[5]);     // Display output?
  int SelfTID, NumTasks;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
  MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);
  if(NumTasks==1){
  printf("Number of tasks = %d\n", NumTasks);
  printf("Size %dx%d with propability for %d Number of Tasks: %0.1f%%\n", N, M, NumTasks, thres*100);
  
  board = NULL;
  newboard = NULL;
  
  board = (int *)malloc(N*M*sizeof(int));

  if (board == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }

  /* second pointer for updated result */
  newboard = (int *)malloc(N*M*sizeof(int));

  if (newboard == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }

  initialize_board (board, N, M);
  printf("Board initialized\n");
  generate_table (board, N, M, thres);
  printf("Board generated\n");

  /* play game of life 100 times */

  for (i=0; i<t; i++) {
    if (disp) display_table (board, N, M);
    play (board, newboard, N, M);    
  }
  printf("Game finished after %d generations.\n", t);
  }
  else if(NumTasks==2){
	  N=N/NumTasks;
	  if(SelfTID==0) printf("Split the Board to %d pieces of (Size/NumofTasks)  %dx%d with propability: %0.1f%%\n", NumTasks, N, M, thres*100 );

	  if(SelfTID==0){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
		   } 
	  
	  else if(SelfTID==1){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
		   }

  }
  else if(NumTasks==4){
 	  N=N/NumTasks;
	  if(SelfTID==0) printf("Split the Board to %d pieces of (Size/NumofTasks) %dx%d with propability: %0.1f%%\n", NumTasks, N, M, thres*100 );
	 
	  
	  if(SelfTID==0){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
	  }
	  else if(SelfTID==1){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
		   
	  }
	  else if(SelfTID==2){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
		   
	  }
	  else if(SelfTID==3){
		   board = NULL;
		   newboard = NULL;
		   
		   board = (int *)malloc(N*M*sizeof(int));

		   if (board == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   /* second pointer for updated result */
		   newboard = (int *)malloc(N*M*sizeof(int));

		   if (newboard == NULL){
		     printf("\nERROR: Memory allocation did not complete successfully!\n");
		     return (1);
		   }

		   initialize_board (board, N, M);
		   printf("Board initialized from task %d\n", SelfTID);
		   generate_table (board, N, M, thres);
		   printf("Board generated from task %d\n", SelfTID);

		   /* play game of life 100 times */

		   for (i=0; i<t; i++) {
		     if (disp) display_table (board, N, M);
		     play (board, newboard, N, M);    
		   }
		   printf("Game finished after %d generations for task %d.\n", t, SelfTID);
		   
	  }	  
  }
  else {
	  printf("Give me the right number of tasks/nodes");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return(0);
}
