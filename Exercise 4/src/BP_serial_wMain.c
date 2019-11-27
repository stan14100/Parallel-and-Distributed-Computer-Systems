// export LD_LIBRARY_PATH=/home/plato/Documents/python_coding/
// compile C programm with gcc -fPIC -shared -o concBP.so BP_serial.c

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#define weights(i, j, k) _weights[i][k+sizes[i]*j]
#define biases(i, j) _biases[i][j]
#define data(i,j) _data[j + i*datadim]
#define res(i,j) _res[j + i*resdim]

#define nabla_w(i, j, k) nabla_w[i][k+sizes[i]*j]
#define nabla_b(i, j) nabla_b[i][j]

#define delta_nabla_w(i, j, k) delta_nabla_w[i][k+sizes[i]*j]
#define delta_nabla_b(i, j) delta_nabla_b[i][j]

//data to transfer to each thread and to BP consequtively
typedef struct{
     int startData, endData, datadim, resdim, Nlayers;
     int *sizes, *res;
     float **weights, **biases, *data;
     //return values : the partial derivates of cost function
     float **delta_nabla_b, **delta_nabla_w;
}BPdata;

//serial matrix mult
void matrix_mult(float **A, int AisTransposed, int Aind, int Arow, int Acol \
     , float ** B, int BisTransposed, int Bind, int Brow, int Bcol \
     , float **C, int Cind )
{
     int i, j, k, l;
     float sum;

     if (Acol != Brow)
          return;

     for(i=0 ; i<Arow ; i++)
     {
          for(j=0 ; j<Bcol; j++)
          {
               C[Cind][j + i*Bcol] = 0;
               if (AisTransposed == 0) //if A is not transposed
               {
                    if (BisTransposed == 0)
                    {
                         for(k=0 ; k<Acol ; k++) //Acol = Brow
                         {
                              C[Cind][j + i*Bcol] += A[Aind][k + i*Acol] * B[Bind][j + k*Bcol] ;
                         }
                    }else
                    {
                         for(k=0 ; k<Acol ; k++) //Acol = Brow
                              C[Cind][j + i*Bcol] += A[Aind][k + i*Acol] * B[Bind][k + j*Brow] ;
                    }
               }else  //isxyei oti transpose(A[i][j]) = A[j][i]
               {
                    if (BisTransposed == 0)
                    {
                         for(k=0 ; k<Acol ; k++) //Acol = Brow
                              C[Cind][j + i*Bcol] += A[Aind][i + k*Arow] * B[Bind][j + k*Bcol] ;
                    }else
                    {
                         for(k=0 ; k<Acol ; k++) //Acol = Brow
                              C[Cind][j + i*Bcol] += A[Aind][i + k*Arow] * B[Bind][k + j*Brow] ;
                    }
               }
          }
     }
}


//function called by threads. implements backPropagation. Here CUDA happens.
void backPropagation(float *_data, int *_res, int ind, int datadim \
     , int resdim, int Nlayers, int *sizes, float **_weights \
     , float **_biases, float **delta_nabla_b, float **delta_nabla_w )
{
     int i, j, k;
     //delta_nabla_b & delta_nabla_w are already initialized


     //build matrices I will need
     //activations matrix. dimensions are as neurons are per Layer
     float **as;
     as = (float **)malloc(sizeof(float*) * Nlayers);
     for (i=0; i<Nlayers; i++)
          as[i] = (float *)malloc(sizeof(float)*sizes[i]);
     //initialize as[0] to data
     // (datadim = sizes[0] kata kanona)
     for (i=0 ; i<sizes[0] ; i++)
          as[0][i] = data(ind,i);

     //zs matrix. same dimensions as activations matrix
     float **zs;
     zs = (float **)malloc(sizeof(float*) * (Nlayers-1));
     for (i=0; i<Nlayers-1; i++)
          zs[i] = (float *)malloc(sizeof(float)*sizes[i+1]);

     //delta matrix. same dimensions as d_nabla_b or biases
     float **delta;
     delta = (float **)malloc(sizeof(float *) * (Nlayers-1));
     for (i=0 ; i<Nlayers-1 ; i++)
          delta[i] = (float *)malloc(sizeof(float)*sizes[i+1]);

     //feedforward
     for (i=0; i<Nlayers-1; i++)
     {
//------------------------MUST IMPLEMENT THIS------------------------//
          //weights * as is a matrix multiplication
          // biases[i][] as a column vector sizes[i+1] x 1
          //weights[i][][] as a sizes[i+1] x sizes[i]
          //as[i][] as column vector sizes[i] x 1
          //result of matrix mult is vector sizes[i+1] x 1
          //zs[i] = CUDA{ weights[i][][] * as[i][] + biases[i][] } ;
          matrix_mult(_weights, 0, i, sizes[i+1], sizes[i], as, 0, i, sizes[i], 1 ,zs, i);

          //as[i+1] = sigmoid(zs[i]) ;
          for (j=0; j<sizes[i+1]; j++) //for every element add the bias and do the sigmoid
          {
               //add biases
               zs[i][j] += biases(i, j);
               //do the sigmoid
               as[i+1][j] = 1.0/(1.0 + exp(-zs[i][j]) );
          }
          //sigmoid(z) = 1.0/(1.0 + exp(z))
     }

     //feedforward finished. Just calculated all z & activations

     //now calcualte partial derivatives of Cost function
     //based on equations BP1 BP2 BP3 BP4

     //last value of zs and delta matrices
     i = Nlayers-2;
     //the correspondig value for activation matrix as is (i+1)

     //BP1 equation : pseudocode
     //delta[i] = cost_function_derivative(as[i+1][],res[index][]) \
     //     * sigmoid_derivative(zs[i]) ;
     //sigmoid_derivative(z) = sigmoid(z)*(1-sigmoid(z))

     for(j=0 ; j<sizes[i+1]; j++)
     {
          //BP1
          delta[i][j] = ( as[i+1][j]-res(ind,j) ) *  \
               ( (1.0/(1.0+exp(-zs[i][j]) ) ) * (1 - (1.0/(1.0+exp(-zs[i][j]) ) ) ) ) ;
          //BP3
          delta_nabla_b(i, j) = delta[i][j] ;
     }
     //BP4 matrix multiplication
     //delta[i] as column sizes[i+1] x 1 and as[i] as row 1 x sizes[i]
     //will result in an array sizes[i+1] x sizes[i]
     //, which are the dimensions for d_nabla_w[i][][]
//------------------------MUST IMPLEMENT THIS------------------------//
     //delta_nabla_w[i][] = delta[i] * as[i]
     matrix_mult(delta, 0, i, sizes[i+1], 1, as, 0, i, 1, sizes[i], delta_nabla_w, i );

     for(i=i-1 ; i>=0 ; i--) //go backwards
     {
          //BP2 matrix multiplication
          //weights[i+1][][] is a sizes[i+2] x sizes[i+1] matrix
          //so transpose(weights[i+1][][]) is a sizes[i+1] x sizes[i+2]
          //delta[i+1][] is a column vector sizes[i+2] x 1
          //result is sizes[i+1] x 1, as delta[i] must be
//------------------------MUST IMPLEMENT THIS------------------------//
          //delta[i][] = ( transpose(weights[i+1][][]) * delta[i+1][] )
          matrix_mult(_weights, 1, i+1, sizes[i+1], sizes[i+2], delta, 0, i+1, sizes[i+2], 1, delta, i);

          //do the Hadamard product (...BP2)
          for(j=0 ; j<sizes[i+1] ; j++)
          {
               delta[i][j] = delta[i][j] * \
               ((1.0/(1.0+exp(-zs[i][j]) ) ) * (1 - (1.0/(1.0+exp(-zs[i][j]) ) ) ) ) ;

               //BP3
               delta_nabla_b(i, j) = delta[i][j];
          }
          //BP4 matrix multiplication
          //delta[i][] is a sizes[i+1] x 1 and transpose(as[i]) is a 1 x sizes[i]
          //so result is sizes[i+1] x sizes[i] ,as must be
//------------------------MUST IMPLEMENT THIS------------------------//
          //delta_nabla_w[i][] = delta[i] * transpose(as[i])
          matrix_mult(delta, 0, i, sizes[i+1], 1, as, 1, i, 1, sizes[i], delta_nabla_w, i);
     }
}

//function executed by threads
void *callBP(void* _bpd)
{
     int ind, i, j, k;
     float **sum_b, **sum_w;
     BPdata *bpd = (BPdata*) _bpd ;
//tdlt     printf("thread %lu.\t start:%d & end:%d\n", pthread_self(),  bpd->startData, bpd->endData);
     //allocate and initialize sum_b & sum_w. ofc they are of same dimensions as weights and biases
     sum_b = (float **)malloc(sizeof(float *) * (bpd->Nlayers-1));
     sum_w = (float **)malloc(sizeof(float *) * (bpd->Nlayers-1));
     for (i=0 ; i<bpd->Nlayers-1 ; i++)
     {
          sum_b[i] = (float *)malloc(sizeof(float)*bpd->sizes[i+1]);
          //allocated. now initialize
          for (j=0 ; j<bpd->sizes[i+1] ; j++)
          {
               //initialize
               sum_b[i][j] = 0;
          }

          sum_w[i] = (float *)malloc(sizeof(float)*(bpd->sizes[i+1]*bpd->sizes[i]) );
          //allocated. now initialize
          for (j=0; j<bpd->sizes[i+1] ; j++)
          {
               for (k=0 ; k <bpd->sizes[i] ; k++)
               {
                    //initialize
                    sum_w[i][k+bpd->sizes[i]*j] = 0;
               }
          }
     }

     //iterate through the data that was assigned to this thread
     int count = 0;
     for(ind=bpd->startData; ind<=bpd->endData; ind++ )
     {
          count++;
          //initialize delta_nabla_b & delta_nabla_w to zero
          for (i=0 ; i<bpd->Nlayers-1 ; i++)
          {
               //delta_nabla_b
               for (j=0 ; j<bpd->sizes[i+1] ; j++)
                    bpd->delta_nabla_b[i][j] = 0;

               //delta_nabla_w
               for (j=0; j<bpd->sizes[i+1] ; j++)
               {
                    for (k=0 ; k <bpd->sizes[i] ; k++)
                         bpd->delta_nabla_w[i][k+bpd->sizes[i]*j] = 0;
               }
          }

          //do BP
          backPropagation(bpd->data, bpd->res, ind, bpd->datadim \
               , bpd->resdim, bpd->Nlayers, bpd->sizes, bpd->weights \
               , bpd->biases, bpd->delta_nabla_b, bpd->delta_nabla_w );
          //backPropagation gave value to delta_nabla_b & delta_nabla_w
          //add delta_nabla_w & delta_nabla_b to sum_w and sum_b correspondingly
          for (i=0 ; i<bpd->Nlayers-1 ; i++)
          {
               //delta_nabla_b
               for (j=0 ; j<bpd->sizes[i+1] ; j++)
                    sum_b[i][j] = sum_b[i][j] + bpd->delta_nabla_b[i][j];

               //delta_nabla_w
               for (j=0; j<bpd->sizes[i+1] ; j++)
               {
                    for (k=0 ; k <bpd->sizes[i] ; k++)
                         sum_w[i][k+bpd->sizes[i]*j] = sum_w[i][k+bpd->sizes[i]*j] + bpd->delta_nabla_w[i][k+bpd->sizes[i]*j] ;
               }
          }
          //return sum_w & sum_b (as delta_nabla_w & delta_nabla_b correspondingly)
     }
     bpd->delta_nabla_w = sum_w;
     bpd->delta_nabla_b = sum_b;
     /* tdlt
     BPdata *bpd = (BPdata*) _bpd ;
     printf("thread %lu.\t Data to do : %d \n", pthread_self(), bpd->endData - bpd->startData + 1);
     //random value to delta_nabla_w & delta_nabla_w
     for (int i=0 ; i< bpd->Nlayers-1 ; i++)
     {
          for (int j=0 ; j<bpd->sizes[i+1] ; j++)
          {
               bpd->delta_nabla_b[i][j] = 43.0/12.0 ;
          }

          for (int j=0; j<bpd->sizes[i+1] ; j++)
          {
               for (int k=0 ; k <bpd->sizes[i] ; k++)
               {
                    bpd->delta_nabla_w[i][k+bpd->sizes[i]*j] = 43.0/12.0;
               }
          }
     }
     */
}

void parallelBP(int Nlayers, float **_weights, float  **_biases, int *sizes
     , float *_data, int *_res,int batchsize, int datadim, int resdim, float eta)
{
     //data transfer happened. code in the end shows how to access it
     //batchsize is the number of data (and res) I have

     //indexes
     int i, j, k, l;
     //get the number of proccesors currently available on the system
     //works probably for Linux. HyperThreading is embedded in the number
     //will be used as Number of Threads too
     const int Ncores = sysconf(_SC_NPROCESSORS_ONLN);
     //declare and initialize nabla_b & nabla_w with, which are
     //the sum of partial derivatives I need for updating biases & weights
     //nabla_b & babla_w have the same dimensions as biases & weighst correspondingly
     float **nabla_w, **nabla_b;
     //will allocate along with the computation (to achieve better computational time)


     //now I need to use backPropagation for each set of data-res exclusively
     //I will break the data in chunks, based on the processors i have (Ncores)
     //Each data chunk will be assigned in a thread.
     //leftOver data will be assigned to the last thread
     BPdata *bpd;
     bpd = (BPdata *)malloc(sizeof(BPdata));

     int startData, endData;

     bpd->datadim = datadim;
     bpd->resdim = resdim;
     bpd->Nlayers = Nlayers;
     bpd->sizes = sizes;
     bpd->weights = _weights;
     bpd->biases = _biases;
     //give pointer to whole data, but will act on restricted
     //based on startData and endData indexes
     bpd->data = _data;
     bpd->res = _res;
     //allocate delta_nabla_b & delta_nabla_w. These are the return values I want
     bpd->delta_nabla_b = (float **)malloc(sizeof(float *) * (bpd->Nlayers-1));
     bpd->delta_nabla_w = (float **)malloc(sizeof(float *) * (bpd->Nlayers-1));
     for (j=0 ; j<bpd->Nlayers-1 ; j++)
     {
          bpd->delta_nabla_b[j] = (float *)malloc(sizeof(float)*bpd->sizes[j+1]);
          //allocated.
          bpd->delta_nabla_w[j] = (float *)malloc(sizeof(float)*(bpd->sizes[j+1]*bpd->sizes[j]) );
          //allocated
     }
     bpd->startData = 0;
     bpd->endData = batchsize-1;

     //begin BP
     callBP((void *)bpd);


     //allocate and compute nabla_b & nabla_w
     nabla_b = (float **)malloc(sizeof(float *) * (Nlayers-1));
     nabla_w = (float **)malloc(sizeof(float *) * (Nlayers-1));
     for (i=0 ; i<Nlayers-1 ; i++)
     {
          nabla_b[i] = (float *)malloc(sizeof(float)*sizes[i+1]);
          //allocated. now compute
          for (j=0 ; j<sizes[i+1] ; j++)
               nabla_b(i, j) = bpd->delta_nabla_b[i][j];

          nabla_w[i] = (float *)malloc(sizeof(float)*(sizes[i+1]*sizes[i]) );
          //allocated. now compute
          for (j=0; j<sizes[i+1] ; j++)
          {
               for (k=0 ; k <sizes[i] ; k++)
                    nabla_w(i,j,k) = bpd->delta_nabla_w[i][k+sizes[i]*j];
          }
     }

     //biases and weights update
     for (int i=0 ; i<Nlayers-1 ; i++)
     {
          //biases
          for (int j = 0; j<sizes[i+1] ; j++)
               biases(i,j) = biases(i,j) - eta/batchsize * nabla_b(i,j);

          //weights
          for (int j=0; j<sizes[i+1] ; j++)
          {
               for (int k=0 ; k <sizes[i] ; k++)
                    weights(i,j,k) = weights(i,j,k) - eta/batchsize * nabla_w(i,j,k);
          }
     }
     //finished. Biases and Weights updated

     //biases
     /*
     for (int i=0 ; i<Nlayers-1 ; i++)
     {
          //oi diastaseis ksekinoyn apo ena layer pio panw
          printf("\nNo%d biases array with %d neurons:\n",i , sizes[i+1] );
          for (int j = 0; j<sizes[i+1] ; j++)
               printf("%.8f ", biases(i,j) );
     }
     //weights
     for (int i=0 ; i<Nlayers-1 ; i++)
     {
          printf("\n\nNo%d weight Array with %d elements",i , sizes[i+1]*sizes[i]);
          for (int j=0; j<sizes[i+1] ; j++)
          {
               printf("\nRow %d: ", j);
               for (int k=0 ; k <sizes[i] ; k++)
                    printf("%f ", weights(i,j,k) );
          }
     }
     */
}


int main(int argc, char *argv[])
{
     int Nlayers, batchsize, datadim, resdim, *_res, *sizes , i, j, k;
     float eta, **_biases, **_weights, *_data, randMax ;

     if (argc < 5)
     {
          printf("Error. Use the following format to execute :\n" );
          printf("Syntax : ./*.out ETA BATCHSIZE Layer1_neurons Layer2_neurons Layer3_neurons . . . \n" );
          return 0;
     }
     eta = strtof(argv[1], NULL) ;
     batchsize = atoi(argv[2]) ;
     datadim = 784;
     resdim =10;

     sizes = (int*) malloc (sizeof(int) * (argc-3));

     Nlayers = 0;
     for (i=3 ; i<argc; i++)
     {
          sizes[i-3] = atoi(argv[i]);
          Nlayers++;
     }

     printf("eta : %f\n", eta );
     printf("batchsize : %d\n", batchsize );
     for (i=0 ; i<Nlayers; i++)
          printf("sizes[%d]: %d\n",i, sizes[i] );

     //fill arrays with random numbers
//     srand(time(NULL));   // should only be called once
     srand(9807);
     randMax = 5.0;

     _biases = (float **)malloc(sizeof(float *) * (Nlayers-1));
     _weights = (float **)malloc(sizeof(float *) * (Nlayers-1));
     for (i=0 ; i<Nlayers-1 ; i++)
     {
          _biases[i] = (float *)malloc(sizeof(float)*sizes[i+1]);
          //allocated. now compute
          for (j=0 ; j<sizes[i+1] ; j++)
               biases(i, j) = (float)rand()/(float)(RAND_MAX/randMax);

          _weights[i] = (float *)malloc(sizeof(float)*(sizes[i+1]*sizes[i]) );
          //allocated. now compute
          for (j=0; j<sizes[i+1] ; j++)
          {
               for (k=0 ; k <sizes[i] ; k++)
                    weights(i,j,k) = (float)rand()/(float)(RAND_MAX/randMax);
          }
     }

     //allocate data
     _data = (float*)malloc(sizeof(float)*(datadim*batchsize));
     _res = (int*)malloc(sizeof(int)*(resdim*batchsize));
     //init data
     for (i=0 ; i<batchsize ; i++)
     {
          for (j=0 ; j<datadim ; j++)
               data(i,j) = (float)rand()/(float)(RAND_MAX/randMax);
     }
     //and res
     for (i=0 ; i<batchsize ; i++)
     {
          for (j=0 ; j<resdim ; j++)
               res(i,j) = rand();
     }

     // measuring compute time
     struct timeval startwtime, endwtime;
     gettimeofday (&startwtime, NULL);

     parallelBP(Nlayers, _weights, _biases, sizes
          , _data, _res, batchsize, datadim, resdim, eta);

     //see time that has been spared
     gettimeofday (&endwtime, NULL);
     double hash_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                   /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
     printf("Time needed (serial) : %fsec\n", hash_time);

/*
     //biases
//     for (int i=0 ; i<Nlayers-1 ; i++)
//     {
     i = Nlayers-2;
          //oi diastaseis ksekinoyn apo ena layer pio panw
          printf("\nNo%d biases array with %d neurons:\n",i , sizes[i+1] );
          for (int j = 0; j<sizes[i+1] ; j++)
               printf("%.8f ", biases(i,j) );

//     }
     //weights
//     for (int i=0 ; i<Nlayers-1 ; i++)
//     {
     i = Nlayers-2;
          printf("\n\nNo%d weight Array with %d elements",i , sizes[i+1]*sizes[i]);
          for (int j=0; j<sizes[i+1] ; j++)
          {
               printf("\nRow %d: ", j);
               for (int k=0 ; k <sizes[i] ; k++)
                    printf("%f ", weights(i,j,k) );
          }
//     }
     printf("\n");
*/


}


//----------------- HOW TO ACCESS DATA PASSED FROM PYTHON ----------------- //

/*
     //single data matrix
     for (int j=0 ; j<datadim ; j++)
     {
          printf("data[1000][%d]:%f\n", j, data(1000,j) );
     }
*/
/*
     //data
     for (int i=0 ; i<batchsize ; i++)
     {
          for (int j=0 ; j<datadim ; j++)
          {
               printf("data[%d][%d]:%f\n",i,j, data(i,j) );
          }
     }

*/
/*
     //res
     for (int i=0 ; i<batchsize ; i++)
     {
          printf("row[%d] : ", i);
          for (int j=0 ; j<resdim ; j++)
          {
               //printf("res[%d][%d]:%d\n",i,j, res(i,j) );
               printf("%d ", res(i,j) );
          }
          printf("\n");
     }
*/

/*
     //biases
     for (int i=0 ; i<Nlayers-1 ; i++)
     {
          //oi diastaseis ksekinoyn apo ena layer pio panw
          printf("New row with %d elements\n", sizes[i+1] );
          for (int j = 0; j<sizes[i+1] ; j++)
          {
               //biases[i][j] = i*100 + j ;
               printf("biases : %.8f\n", biases(i,j) );
          }
     }
*/
/*
     //weights
     for (int i=0 ; i<Nlayers-1 ; i++)
     {
          printf("Next weights array with %d elements\n", sizes[i+1]*sizes[i]);

     //     for (int j=0; j<sizes[i+1]*sizes[i] ; j++)
     //     {
     //          printf("weights %f\n", weightss[i][j]);
     //     }
          int offset = 0;
          for (int j=0; j<sizes[i+1] ; j++)
          {
               printf("Next row\n");
               for (int k=0 ; k <sizes[i] ; k++)
               {
                    printf("weightss :%f\n", _weights[i][offset]);
                    offset++;
                    printf("weights[%d][%d][%d] :%f\n",i,j,k, weights(i,j,k) );
               }
          }
     }
*/
