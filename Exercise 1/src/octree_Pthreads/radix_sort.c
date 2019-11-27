#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include<pthread.h>

#define MAXBINS 8
struct radix_inputs {
    unsigned long int *morton_codes;
    unsigned long int *sorted_morton_codes;
    unsigned int *permutation_vector;
    unsigned int *index;
    int *level_record;
    int N;
    int population_threshold;
    int sft;
    int lv;
    int father;
};

inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}
extern unsigned int NUM_THREADS;
int volatile unsigned active_threads;
pthread_mutex_t active;


void *pthread_radix(void *str){
  struct radix_inputs *child_inputs;
  struct radix_inputs *my_data=(struct radix_inputs *) str;




  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;



  if(my_data->N<=0){
	  if (my_data->father == 1) //an to thread poy teleiwnei einai child tote meiwne
	     		{
	     			pthread_mutex_lock(&active);
	     				active_threads--;
	     			pthread_mutex_unlock(&active);
	     		}
    return;
  }
  else if(my_data->N<=my_data->population_threshold || my_data->sft < 0) { // Base case. The node is a leaf

	  my_data->level_record[0] = my_data->lv; // record the level of the node
    memcpy(my_data->permutation_vector, my_data->index, my_data->N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(my_data->sorted_morton_codes,my_data-> morton_codes, my_data->N*sizeof(unsigned long int)); // Copy the Morton codes
    if (my_data->father == 1) //an to thread poy teleiwnei einai child tote meiwne
       		{ 
       			pthread_mutex_lock(&active);
       				active_threads--;
       			pthread_mutex_unlock(&active);
       		}
    return;
  }
  else{

	  my_data->level_record[0] = my_data->lv;
    // Find which child each point belongs to
    for(int j=0; j<my_data->N; j++){
      unsigned int ii = (my_data->morton_codes[j]>>my_data->sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }

    for(int j=0; j<my_data->N; j++){
      unsigned int ii = (my_data->morton_codes[j]>>my_data->sft) & 0x07;
      my_data->permutation_vector[BinCursor[ii]] = my_data->index[j];
      my_data->sorted_morton_codes[BinCursor[ii]] = my_data->morton_codes[j];
      BinCursor[ii]++;
    }

    //swap the index pointers
    swap(&my_data->index, &my_data->permutation_vector);

    //swap the code pointers
    swap_long(&my_data->morton_codes, &my_data->sorted_morton_codes);
    child_inputs=(struct radix_inputs *)malloc(MAXBINS*sizeof(struct radix_inputs));
    /* Call the function recursively to split the lower levels */
    for(int i=0; i<MAXBINS; i++){
      int offset = (i>0) ? BinSizes[i-1] : 0;
      int size = BinSizes[i] - offset;
      child_inputs[i].morton_codes=&my_data->morton_codes[offset];
	  child_inputs[i].sorted_morton_codes=&my_data->sorted_morton_codes[offset];
	  child_inputs[i].permutation_vector=&my_data->permutation_vector[offset];
	  child_inputs[i].index=&my_data->index[offset];
	  child_inputs[i].level_record=&my_data->level_record[offset];
	  child_inputs[i].N=size;
	  child_inputs[i].population_threshold=my_data->population_threshold;
	  child_inputs[i].sft=my_data->sft-3;
	  child_inputs[i].lv=my_data->lv+1;
	  pthread_mutex_lock(&active);
	  if(active_threads<NUM_THREADS){
		  active_threads++;
		  pthread_mutex_unlock(&active);
                  pthread_t newthread;
		  child_inputs[i].father=1;
		  pthread_attr_t detached;
		  pthread_attr_init(&detached);
		  pthread_attr_setdetachstate(&detached, PTHREAD_CREATE_DETACHED);
		  pthread_create(&newthread,&detached,pthread_radix,(void *)(child_inputs+i));


	  }
	  else {
		  pthread_mutex_unlock(&active);
		  child_inputs[i].father=0;
		  pthread_radix((void *) (child_inputs+i));

	  }

    }
    if (my_data->father == 1) //an to thread poy teleiwnei einai child tote meiwne
    		{ 
    			pthread_mutex_lock(&active);
    				active_threads--;
    			pthread_mutex_unlock(&active);
    		}

  }
}
void truncated_radix_sort(unsigned long int *morton_codes,
			  unsigned long int *sorted_morton_codes,
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N,
			  int population_threshold,
			  int sft, int lv){
	pthread_mutex_init(&active, NULL);
	struct radix_inputs *master_inputs;
	master_inputs=(struct radix_inputs *)malloc(sizeof(struct radix_inputs));
	master_inputs->morton_codes=morton_codes;
	master_inputs->sorted_morton_codes=sorted_morton_codes;
	master_inputs->permutation_vector=permutation_vector;
	master_inputs->index=index;
	master_inputs->level_record=level_record;
	master_inputs->N=N;
	master_inputs->population_threshold=population_threshold;
	master_inputs->sft=sft;
	master_inputs->lv=lv;
	master_inputs->father=0;
	active_threads=0;
   
    pthread_radix((void *) master_inputs);
    while (active_threads > 0);{}
	pthread_mutex_destroy(&active);
    free(master_inputs);
}
