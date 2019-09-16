#include <stdio.h>
#include "timerc.h"

__global__ void warmup(){

}


__global__ void finishCumSum(int *input, int sizeI, int* output, int sizeO){

	int numElementsPerBlock = sizeI/sizeO;

        int* s_input = input + numElementsPerBlock*(blockIdx.x + 1) ;

	s_input[threadIdx.x] +=  output[blockIdx.x];
        s_input[threadIdx.x + numElementsPerBlock/2 ] += output[blockIdx.x];

}

__global__ void vectorCumSum(int *input, int size, int* output, int WO){
	int numElementsPerBlock = size/gridDim.x;
	int* s_input = input + numElementsPerBlock*blockIdx.x;
	for(int s = 1; s<=numElementsPerBlock/2; s=s*2){
		if( threadIdx.x < numElementsPerBlock/(2*s)){
			s_input[threadIdx.x*2*s + s-1+s] = s_input[threadIdx.x*2*s + s-1] + s_input[threadIdx.x*2*s+s-1 + s];

		}
		__syncthreads();
	}

   	for(int s = numElementsPerBlock/4; s >= 1; s=s/2){
                if( threadIdx.x < -1 + numElementsPerBlock/(2*s)){
                        s_input[threadIdx.x*2*s + 2*s-1+s] = s_input[threadIdx.x*2*s + 2*s-1] + s_input[threadIdx.x*2*s + 2*s-1 + s];

                }
                __syncthreads();
        }

	if (WO == 1)
		output[blockIdx.x] = s_input[numElementsPerBlock - 1]; 

}


__global__ void vectorSumBetterCoalescedWithPresum(int *input,int c_size, int size, int *output){
        int numElementsPerBlock = size/(gridDim.x*c_size);

        int total1 = 0;
        int total2 = 0;

        for(int i = 0; i < c_size;i++){
                total1 = total1 + input[i*2*blockDim.x + threadIdx.x + blockIdx.x*numElementsPerBlock*c_size];
                total2 = total2 + input[i*2*blockDim.x + blockDim.x + threadIdx.x + blockIdx.x*numElementsPerBlock*c_size];
	}

        __shared__ int s_input[2048];
        s_input[threadIdx.x] = total1;
        s_input[threadIdx.x + (numElementsPerBlock/2) ] = total2;
        __syncthreads();

        for(int s = numElementsPerBlock/2; s>=1; s=s/2){
                if( threadIdx.x < s){
                        s_input[threadIdx.x] = s_input[threadIdx.x] + s_input[threadIdx.x+s];

                }
                __syncthreads();
        }
        output[blockIdx.x] = s_input[0];
}



__global__ void vectorSumBetterCoalesced(int *input, int size, int *output){
        int numElementsPerBlock = size/gridDim.x;

        __shared__ int s_input[2048];
        s_input[threadIdx.x] = input[threadIdx.x + blockIdx.x*numElementsPerBlock];
        s_input[threadIdx.x + (numElementsPerBlock/2) ] = input[threadIdx.x + (numElementsPerBlock/2) + blockIdx.x];
        __syncthreads();

        for(int s = numElementsPerBlock/2; s>=1; s=s/2){
                if( threadIdx.x < s){
                        s_input[threadIdx.x] = s_input[threadIdx.x] + s_input[threadIdx.x+s];

                }
                __syncthreads();
        }
        output[blockIdx.x] = s_input[0];
}




__global__ void vector_sum_naive(int *input, int size, int *output){
        int abs_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int numElementsPerThread = size / ( gridDim.x * blockDim.x);
        int startPos = abs_thread_idx * numElementsPerThread;
        int localTotal = 0;

        for(int i = 0; i < numElementsPerThread; i++){
                localTotal = localTotal + input[i + startPos];
        }

	output[abs_thread_idx] = localTotal;

}

__global__ void vector_sum(int *input, int size, int *output){
	int abs_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int numElementsPerThread = size / (blockDim.x * gridDim.x);
	int startPos = abs_thread_idx * numElementsPerThread;
	int localTotal = 0;

	for(int i = 0; i < numElementsPerThread; i++){
		localTotal = localTotal + input[i + startPos]; 
	}
	
	__shared__ int totals[2048];
	totals[abs_thread_idx] = localTotal;
	__syncthreads();	

	if(abs_thread_idx == 0){
		for(int i = 1; i < blockDim.x * gridDim.x; i++){			
			localTotal = localTotal + totals[i];
		}
		output[0] = localTotal;
	}

}

int main(){
	int numElements = 128*1024 * 1024;
	int *hostInput = (int *) malloc(numElements * sizeof(int));
	for(int i = 0; i < numElements; i++){
		hostInput[i] = 1;
	}
	int *deviceInput;
	int *deviceOutput;
	int hostOutput[1024*1024];

	int cpu_total = 0;
	float cpu_time;
	cstart();
	for (int i = 0; i < 128*1024*1024; i++){
		cpu_total = cpu_total + 1;
	}
	cend(&cpu_time);
	printf("Cpu total = %d\n", cpu_total);
        printf("Cpu time = %f\n", cpu_time);

	warmup<<<1,1>>>();

	float malloc_and_cpy_time;
	gstart();
	cudaMalloc((void **) &deviceInput, numElements * sizeof(int));
	cudaMalloc((void **) &deviceOutput, 128*1024 * 1024 * sizeof(int));
	cudaMemcpy(deviceInput, hostInput, numElements* sizeof(int),cudaMemcpyHostToDevice);
	gend(&malloc_and_cpy_time);	

	
	float CumSumKernelCallTime;
	gstart();
        vectorCumSum<<<512*128 , 1024>>>(deviceInput, 128*1024*1024, deviceOutput, 1);
	//vectorCumSum<<<1,256*128>>>(deviceOutput, 512*128, NULL, 0);//deviceInput is not needed here
	cudaMemcpy(hostOutput, deviceOutput, 128*512*sizeof(int),cudaMemcpyDeviceToHost);
	int tmpCumSum = 0;
	for (int i = 0; i < 128*512; i++){
                tmpCumSum += hostOutput[i];
                hostOutput[i] = tmpCumSum;
        }
        cudaMemcpy( deviceOutput, hostOutput, 128*512*sizeof(int),cudaMemcpyHostToDevice);

	finishCumSum<<<512*128 - 1 , 1024>>>(deviceInput, 128*1024*1024, deviceOutput, 512*128);        
	gend(&CumSumKernelCallTime);
	
	float CumSumHostTime;
	cstart();
	int hostCumSum = 0;
	for (int i = 0; i < 128*1024*1024; i++){
		hostCumSum += hostInput[i];
		hostInput[i] = hostCumSum; 
	}
	cend(&CumSumHostTime);

        cudaMemcpy(hostInput, deviceInput, 128*1024*1024*sizeof(int),cudaMemcpyDeviceToHost);


        printf("Time Cum Sum CPU time = %f\n", CumSumHostTime);
	printf("Time Cum Sum Kernel Call time = %f\n", CumSumKernelCallTime);
        return 0;


	float naive_gpu_time;
        gstart();
	vector_sum_naive<<<64*4,64/2>>>(deviceInput, 128*1024*1024, deviceOutput);	
	gend(&naive_gpu_time);
	printf("Naive kernel time = %f\n",naive_gpu_time);	

	cudaMemcpy(hostOutput, deviceOutput, 64*64*2 * sizeof(int),cudaMemcpyDeviceToHost);
	int naive_total = 0;
	for (int i = 0; i < 64*64*2; i++){
		naive_total = naive_total + hostOutput[i];
	}
	printf("Total = %d\n", naive_total);
	

	float better_gpu_time;
	gstart();
	vectorSumBetterCoalescedWithPresum<<<256, 1024>>>(deviceInput,256, 128*1024*1024, deviceOutput);
	gend(&better_gpu_time);
        printf("Better kernel time with presum = %f\n",better_gpu_time);

	float copy_back_and_finish_time;
	gstart();
	cudaMemcpy(hostOutput, deviceOutput, 256 * sizeof(int),cudaMemcpyDeviceToHost);
        int better_total = 0;
        for (int i = 0; i < 256; i++){
                better_total = better_total + hostOutput[i];
        }
	gend(&copy_back_and_finish_time);
        
	printf("Total = %d\n", better_total);
	printf("Time to malloc and copy = %f, time to cpy back and finish = %f\n",malloc_and_cpy_time,copy_back_and_finish_time);

	cudaDeviceSynchronize();

}
