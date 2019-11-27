/* #ifndef UTILS_H_   /\* Include guard *\/ */
/* #define UTILS_H_ */

#define Board(x,y) board[(x)*M + (y)]
#define NewBoard(x,y) newboard[(x)*M + (y)]

/* set everthing to zero */

void initialize_board (int *board, int N,int M);

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N);

/* add to a height index, wrapping around */

int yadd (int i, int a, int M);

/* return the number of on cells adjacent to the i,j cell */

int adjacent_to (int *board, int i, int j, int N,int M);

/* play the game through one generation */

void play (int *board, int *newboard, int N,int M);

/* print the life board */

void print (int *board, int N,int M);

/* generate random table */

void generate_table (int *board, int N,int M,float threshold);

/* display the table with delay and clear console */

void display_table(int *board, int N,int M);

/* #endif // FOO_H_ */
