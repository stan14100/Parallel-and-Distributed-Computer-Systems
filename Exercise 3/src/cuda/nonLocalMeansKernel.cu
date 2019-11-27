#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define img(i,j) A[(i) + (j)*(Nthreads+(2*stop))]
#define g(i) gaussian[(i)]
#define	W(i,j) D[(i) + (j)*(Nthreads)]
//#define pix(i,j) pixel[(i) + (j)*(NumberOfNeighbours)]
//__device__ const int Block=49
#define sh(i,j) shared[(i)+(j)*(16+2*stop)]
__global__ void nonLocalMeansKernel(float const * const A, float const * const gaussian, int NumberOfNeighbours, float filtsigma, int Nthreads, float *D )
{	
	int NoN = NumberOfNeighbours;
	int stop=(NoN/2);
	int i=blockIdx.x * blockDim.x + threadIdx.x + stop;
	int j= blockIdx.y * blockDim.y + threadIdx.y + stop;
	int nbonbhds= Nthreads;
	float sumw=0;
	//int size=5*5;
	//float pixel[];
	extern __shared__ float shared[];
	


	if (i < nbonbhds+stop && j<nbonbhds+stop )
	{
		/*for(int k=-stop; k<=stop; k++)
		{
			for(int z=-stop; z<=stop; z++)
			{
				pix(k+stop,z+stop) = img(i+k,j+z);
			}
		} */
		
		float sigma = filtsigma;
		float distance=0;
		float finaldistance=0;
		float w=0;
		int Blockstop=(nbonbhds)/16;
		for(int countBlockx=0; countBlockx<Blockstop; countBlockx++)
		{
				
			for(int countBlocky=0; countBlocky<Blockstop; countBlocky++)
			{
				if((threadIdx.x<(16+(2*stop))) && (threadIdx.y<(16+(2*stop))))
				{
					sh(threadIdx.x, threadIdx.y) = img(threadIdx.x+(countBlockx*(16)), threadIdx.y+(countBlocky*(16))); 
				}
				__syncthreads();

				for(int n=stop; n<16+stop; n++)
				{
					for(int l=stop; l<16+stop; l++)
					{	
						w=0;
						int count=0;
						distance=0;
						for(int k=-stop; k<=stop; k++)
						{		
							for(int z=-stop; z<=stop; z++)
							{
								distance += g(count)*(img(i+k, j+z) - sh(n+k,l+z))*(img(i+k, j+z) - sh(n+k,l+z));
								count++;

							}
						}
						w = exp(-distance/sigma)*sh(n,l);
						finaldistance+=exp(-distance/sigma);
						sumw += w;
						 
					}
				}
				 
				__syncthreads();
			}	
		}
		W(i-stop,j-stop) = sumw/finaldistance;
			
	}
} 



 



