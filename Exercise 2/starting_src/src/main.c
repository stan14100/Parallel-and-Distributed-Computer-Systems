/*
* Game of Life implementation based on
* http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
*
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <game-of-life.h>
#include <mpi.h>

int main (int argc, char *argv[]) {
	if (argc != 4) { // Check if the command line arguments are correct
		printf("Usage: %s N thres disp\n"
		"where\n"
		"  thres : propability of alive cell\n"
		"  t     : number of generations\n"
		"  disp  : {1: display output, 0: hide output}\n"
		, argv[0]);
		return (1);
	}

	// Input command line arguments
	double thres = atof(argv[1]); // Propability of life cell
	int t = atoi(argv[2]);        // Number of generations
	int disp = atoi(argv[3]);     // Display output?
	int *board, *newboard, i, N, M;
	board = NULL;
	newboard = NULL;
	//int *sb;
//	sb = NULL;


	omp_set_num_threads(8); //set number of threads for omp

	int numtasks,rank; //thats for MPI

	MPI_Init(&argc,&argv); // MPI initialized
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks); //get number of tasks
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);//get my rank


	// *****************************************

	if(numtasks == 1)
	{
		printf("Size 40.000 x 40.000 with propability: %0.1f%%\n",thres*100);
		N=40000;
		M=40000;
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

		initialize_board (board, N,M);
		printf("Board initialized\n");
		generate_table (board, N,M, thres);
		printf("Board generated\n");

		for (i=0; i<t; i++) {
			if (disp) display_table(board, N,M);
			play(board, newboard, N,M);
		}
		printf("Game finished after %d generations.\n", t);

	}



	else if(numtasks == 2)
	{
		//dhmioyrgw shared memory gia kathe task
		N=20000;
		M=80000;
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
		//dhmioyrgw shared memory gia ola ta task


		if(rank==0)
		{
			printf("Size 80.000 x 40.000 with propability: %0.1f%%\n",thres*100);
			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++) {
				if (disp) display_table (board, N,M);
				play (board, newboard, N,M);
			}



		}
		else if(rank==1)
		{
			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++) {
				if (disp) display_table (board, N,M);
				play (board, newboard, N, M);
			}
			printf("Game finished after %d generations.\n", t);
		}

	}else if(numtasks == 4)
	{
		int *board, *newboard,i;
		board = NULL;
		newboard = NULL;
		printf("Size 80.000 x 80.000 with propability: %0.1f%%\n",thres*100);

		N=20000;
		M=80000;
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


		if(rank==0)
		{

			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++) {
				if (disp) display_table (board, N,M);
				play (board, newboard, N,M);
			}



		}
		else if(rank==1)
		{
			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++) {
				if (disp) display_table (board, N,M);
				play (board, newboard, N,M);
			}
		}else if(rank==2)
		{

			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++) {
				if (disp) display_table (board, N,M);
				play (board, newboard, N,M);
			}



		}else if(rank==3)
		{

			initialize_board (board, N,M);
			printf("Board initialized at task: %d\n",rank);
			generate_table (board, N,M, thres);
			printf("Board generated at task: %d\n",rank);

			for (i=0; i<t; i++)
			{
				if (disp) display_table (board, N,M);
				play (board, newboard, N,M);
			}
		}

	}else
	{
		printf("Must specify one two or four processes. Terminating.\n");

	}
	if (rank==0)
		printf("Game finished after %d generations.\n", t);

	MPI_Finalize(); // end of MPI and end of the programm
	return(0);
	// ****************************************
}

//na sbhsw xroynometrhsh toy MPI_Waitall
