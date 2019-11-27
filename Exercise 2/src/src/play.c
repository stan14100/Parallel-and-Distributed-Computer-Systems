#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include<omp.h>
#include<mpi.h>
#include <game-of-life.h>

void play (int *board, int *newboard, int N, int M) {
  /*
    (copied this from some web page, hence the English spellings...)

    1.STASIS : If, for a given cell, the number of on neighbours is 
    exactly two, the cell maintains its status quo into the next 
    generation. If the cell is on, it stays on, if it is off, it stays off.

    2.GROWTH : If the number of on neighbours is exactly three, the cell 
    will be on in the next generation. This is regardless of the cell's
    current state.

    3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will 
    be off in the next generation.
  */
	
	struct timeval startwtime, endwtime;
  int   i, j, a;
  gettimeofday(&startwtime, NULL); 
  /* for each cell, apply the rules of Life */
  int SelfTID, NumTasks;
  MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
  MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);
  MPI_Datatype rowtype;
  MPI_Type_contiguous(M, MPI_INT, &rowtype);
  MPI_Type_commit(&rowtype);
 if(NumTasks==1){   
	 #pragma omp parallel for private(j,a)
	  for (i=0; i<N; i++){  
		for (j=0; j<M; j++) {
		  a = adjacent_to (board, i, j, N, M);
		  if (a == 2) NewBoard(i,j) = Board(i,j);
		  if (a == 3) NewBoard(i,j) = 1;
		  if (a < 2) NewBoard(i,j) = 0;
		  if (a > 3) NewBoard(i,j) = 0;
		}
	  }
 }
 else if(NumTasks==2){
	 MPI_Request req[4];
	 MPI_Status status[4];
	 int Up[M],Down[M];
	 if(SelfTID==0){
		 MPI_Isend(&Board(0,0), 1, rowtype, 1, 2, MPI_COMM_WORLD, &req[0]); //tag=2 gia tis down lwrides (pou 8a katalhksoun)
		 MPI_Isend(&Board(N-1,0), 1, rowtype, 1, 1, MPI_COMM_WORLD, &req[1]);// tag=1 gia tis up lwrides
		 MPI_Irecv(Up, M, MPI_INT, 1, 1, MPI_COMM_WORLD, &req[2]);
		 MPI_Irecv(Down, M, MPI_INT, 1, 2, MPI_COMM_WORLD, &req[3]);
	 }
	 if(SelfTID==1){
		 MPI_Isend(&board[0], 1, rowtype, 0, 2, MPI_COMM_WORLD, &req[0]);
		 MPI_Isend(&board[N-1*(M)], 1, rowtype,0, 1, MPI_COMM_WORLD, &req[1]);
		 MPI_Irecv(Up, M, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[2]);
		 MPI_Irecv(Down, M, MPI_INT, 0, 2, MPI_COMM_WORLD, &req[3]);
	 }
	 
	 #pragma omp parallel for private(j,a)
		 for (i=1; i<N-1; i++){  
			for (j=0; j<M; j++) {
			  a = adjacent_to (board, i, j, N, M);
			  if (a == 2) NewBoard(i,j) = Board(i,j);
			  if (a == 3) NewBoard(i,j) = 1;
			  if (a < 2) NewBoard(i,j) = 0;
			  if (a > 3) NewBoard(i,j) = 0;
			}
		  }
		 
	MPI_Waitall(4, req, status); //Elegxoume kai perimenoume na doume an oles oi entoles send recv teleiwsan gia na sunexisoume an oxi perimenoume mexri na teleiwsoun
								//bebaia exoume krupsei thn ka8usterhsh elegxontas pio prin ton upoloipomeno pinaka 
	//gia thn prwth grammh
	#pragma omp parallel for
	for(j=0; j<M; j++){
		a=0;
		if(j==0){
			a=a+Up[0];
			a=a+Up[M-1];
			a=a+Up[1];
			a=a+Board(0,j+1);
			a=a+Board(0,M-1);
			a=a+Board(1,j);
			a=a+Board(1,j+1);	
			a=a+Board(1,M-1);
		}
		else if(j==M-1){
			a=a+Up[0];
			a=a+Up[M-1];
			a=a+Up[M-2];
			a=a+Board(0,0);
			a=a+Board(0,j-1);
			a=a+Board(1,j);
			a=a+Board(1,j-1);	
			a=a+Board(1,0);
		}
		else{
			a=a+Up[j];
			a=a+Up[j-1];
			a=a+Up[j+1];
			a=a+Board(0,j+1);
			a=a+Board(0,j-1);
			a=a+Board(1,j);
			a=a+Board(1,j+1);	
			a=a+Board(1,j-1);	
		}
		if (a == 2) NewBoard(0,j) = Board(0,j);
		if (a == 3) NewBoard(0,j) = 1;
		if (a < 2) NewBoard(0,j) = 0;
		if (a > 3) NewBoard(0,j) = 0;
	}
	//gia thn teleutaia grammh
	#pragma omp parallel for
		for(j=0; j<M; j++){
			a=0;
			if(j==0){
				a=a+Down[0];
				a=a+Down[M-1];
				a=a+Down[1];
				a=a+Board(N-1,j+1);
				a=a+Board(N-1,M-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j+1);	
				a=a+Board(N-2,M-1);
			}
			else if(j==M-1){
				a=a+Down[0];
				a=a+Down[M-1];
				a=a+Down[M-2];
				a=a+Board(N-1,0);
				a=a+Board(N-1,j-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j-1);	
				a=a+Board(N-2,0);
			}
			else{
				a=a+Down[j];
				a=a+Down[j-1];
				a=a+Down[j+1];
				a=a+Board(N-1,j+1);
				a=a+Board(N-1,j-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j+1);	
				a=a+Board(N-2,j-1);	
			}
			if (a == 2) NewBoard(N-1,j) = Board(N-1,j);
			if (a == 3) NewBoard(N-1,j) = 1;
			if (a < 2) NewBoard(N-1,j) = 0;
			if (a > 3) NewBoard(N-1,j) = 0;
		}
 }
 else if(NumTasks==4){
	 MPI_Request req[4];
	 MPI_Status status[4];
	 int Up[M],Down[M];
	 int giveup,takeup,givedown,takedown;
	 if(SelfTID==0){
		 giveup=3;
		 givedown=1;
		 takeup=3;
		 takedown=1;
	 }
	 else if(SelfTID==1){
		 giveup=0;
		 givedown=2;
		 takeup=0;
		 takedown=2;
	 }
	 else if(SelfTID==2){
		 giveup=1;
		 givedown=3;
		 takeup=1;
		 takedown=3; 
	 }
	 else if(SelfTID==3){
		 giveup=2;
		 givedown=0;
		 takeup=2;
		 takedown=0;
	 }
	 MPI_Isend(&Board(0,0), 1, rowtype, giveup, 2, MPI_COMM_WORLD, &req[0]); //tag=2 gia tis down lwrides (pou 8a katalhksoun)
	 MPI_Isend(&Board(N-1,0), 1, rowtype, givedown, 1, MPI_COMM_WORLD, &req[1]);// tag=1 gia tis up lwrides
	 MPI_Irecv(Up, M, MPI_INT, takeup, 1, MPI_COMM_WORLD, &req[2]);
	 MPI_Irecv(Down, M, MPI_INT, takedown, 2, MPI_COMM_WORLD, &req[3]);
	 #pragma omp parallel for private(j,a)
		 for (i=1; i<N-1; i++){  
			for (j=0; j<M; j++) {
			  a = adjacent_to (board, i, j, N, M);
			  if (a == 2) NewBoard(i,j) = Board(i,j);
			  if (a == 3) NewBoard(i,j) = 1;
			  if (a < 2) NewBoard(i,j) = 0;
			  if (a > 3) NewBoard(i,j) = 0;
			}
		  }
		 
	MPI_Waitall(4, req, status); //Elegxoume kai perimenoume na doume an oles oi entoles send recv teleiwsan gia na sunexisoume an oxi perimenoume mexri na teleiwsoun
								//bebaia exoume krupsei thn ka8usterhsh elegxontas pio prin ton upoloipomeno pinaka 
	//gia thn prwth grammh
	#pragma omp parallel for
	for(j=0; j<M; j++){
		a=0;
		if(j==0){
			a=a+Up[0];
			a=a+Up[M-1];
			a=a+Up[1];
			a=a+Board(0,j+1);
			a=a+Board(0,M-1);
			a=a+Board(1,j);
			a=a+Board(1,j+1);	
			a=a+Board(1,M-1);
		}
		else if(j==M-1){
			a=a+Up[0];
			a=a+Up[M-1];
			a=a+Up[M-2];
			a=a+Board(0,0);
			a=a+Board(0,j-1);
			a=a+Board(1,j);
			a=a+Board(1,j-1);	
			a=a+Board(1,0);
		}
		else{
			a=a+Up[j];
			a=a+Up[j-1];
			a=a+Up[j+1];
			a=a+Board(0,j+1);
			a=a+Board(0,j-1);
			a=a+Board(1,j);
			a=a+Board(1,j+1);	
			a=a+Board(1,j-1);	
		}
		if (a == 2) NewBoard(0,j) = Board(0,j);
		if (a == 3) NewBoard(0,j) = 1;
		if (a < 2) NewBoard(0,j) = 0;
		if (a > 3) NewBoard(0,j) = 0;
	}
	//gia thn teleutaia grammh
	#pragma omp parallel for
		for(j=0; j<M; j++){
			a=0;
			if(j==0){
				a=a+Down[0];
				a=a+Down[M-1];
				a=a+Down[1];
				a=a+Board(N-1,j+1);
				a=a+Board(N-1,M-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j+1);	
				a=a+Board(N-2,M-1);
			}
			else if(j==M-1){
				a=a+Down[0];
				a=a+Down[M-1];
				a=a+Down[M-2];
				a=a+Board(N-1,0);
				a=a+Board(N-1,j-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j-1);	
				a=a+Board(N-2,0);
			}
			else{
				a=a+Down[j];
				a=a+Down[j-1];
				a=a+Down[j+1];
				a=a+Board(N-1,j+1);
				a=a+Board(N-1,j-1);
				a=a+Board(N-2,j);
				a=a+Board(N-2,j+1);	
				a=a+Board(N-2,j-1);	
			}
			if (a == 2) NewBoard(N-1,j) = Board(N-1,j);
			if (a == 3) NewBoard(N-1,j) = 1;
			if (a < 2) NewBoard(N-1,j) = 0;
			if (a > 3) NewBoard(N-1,j) = 0;
		}
 }
 
  /* copy the new board back into the old board */
#pragma omp parallel for private(j)
  for (i=0; i<N; i++)
    for (j=0; j<M; j++) {
      Board(i,j) = NewBoard(i,j);
    }
  gettimeofday(&endwtime, NULL);
  double play_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
  				/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
      
     printf("Time to compute the play codes for task %d  : %fs\n", SelfTID, play_time);
}
