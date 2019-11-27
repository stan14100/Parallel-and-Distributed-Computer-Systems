#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define img(i,j) A[(i) + (j)*(Nthreads+(2*stop))]
#define g(i) gaussian[i]
#define	W(i,j) D[(i) + (j)*Nthreads]
#define pix(i,j) pixel[(i) + (j)*sqrt(NumberOfNeighbours)]
__global__ void nonLocalMeansKernel(float const * const A, float const * const gaussian, int NumberOfNeighbours, int filtsigma, int Nthreads, float *D )
{	
	int NoN = NumberOfNeighbours;
	int stop= sqrt((float)NoN)/2;
	int i=blockIdx.x * blockDim.x + threadIdx.x + stop;
	int j= blockIdx.y * blockDim.y + threadIdx.y + stop;
	int nbonbhds= Nthreads;
	float sumw=0;
	extern __shared__ float shared[];  
	float pixel[NumberOfNeighbours]; 
	
	if (i < nbonbhds+stop && j<nbonbhds+stop )
	{
		for(int k=-stop; k<stop; k++)
		{
			for(int z=-stop; z<stop; z++)
			{
				pix(k+stop,z+stop) = img(i+k,j+z);
			}
		}
	}
		if (i < nbonbhds+stop && j<nbonbhds+stop )
		{
		
			float sigma = filtsigma;
			float distance=0;
			float finaldistance=0;
			float w=0;
			for(int n=stop; n<nbonbhds+stop; n++)
			{
				for(int l=stop; l<nbonbhds+stop; l++)
				{	
					w=0;
					int count=0;
					distance=0;
					for(int k=-stop; k<=stop; k++)
					{	
						for(int z=-stop; z<=stop; z++)
						{
								distance += g(count)*
								count++;

						}
					}
						w = exp(-distance/sigma)*img(n,l);
						finaldistance+=exp(-distance/sigma);
						sumw += w;
					 
	
				}
			}	
		
			W(i-stop,j-stop) = sumw/finaldistance;
				
		}
}   




