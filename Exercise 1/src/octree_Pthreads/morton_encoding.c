#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include<pthread.h>

#define DIM 3
struct morton {
	unsigned long int *morton_codes;
	unsigned int *hash_codes;
	int N;
	int maxlev;
	int NUM_THREADS;
	int id;
};
//extern struct morton morton_inputs;

inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

/* The function that transform the morton codes into hash codes */ 
void *morton_encoding(void *inputs){
	struct morton *my_data;
	   my_data=(struct morton *) inputs;
	   int a=(int) (my_data->N/my_data->NUM_THREADS);
	    int start=(my_data->id)*(a),stop=(my_data->id+1)*(a);
	    if(my_data->id==(my_data->NUM_THREADS-1)) stop=my_data->N;
  for(int i=start; i<stop; i++){
    // Compute the morton codes from the hash codes using the magicbits mathod
    my_data->morton_codes[i] = mortonEncode_magicbits(my_data->hash_codes[i*DIM], my_data->hash_codes[i*DIM + 1], my_data->hash_codes[i*DIM + 2]);
  }
  pthread_exit(NULL);
}


