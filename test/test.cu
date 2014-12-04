
#include <cstdio>
#include <helper_math.h>

//texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex; 
//texture<int, 1, cudaReadModeElementType> transferTex; 


texture<float,  1, cudaReadModeElementType> texture_float_1D;
extern "C" {
__global__ void helloWorld(unsigned int *data1, unsigned int *data2, float *data3, unsigned int width) {
	/*
		int tid, tx, ty;
		tx = blockDim.x*blockIdx.x + threadIdx.x;
		ty = blockDim.y*blockIdx.y + threadIdx.y;
		tid = width*ty + tx;
		
		if ((tx >= width) || (tx >= width)) return;
		
		int Value = 0;
	
		
		for (int i = 0; i < width; i++)
		{
		  int MVal=data1[ty * width + i];
		  int NVal=data2[i * width + tx];
		   Value += MVal * NVal;
		 
		}
		
		data3[tid]= Value;
		*/
		
		
		for(int i = 0; i<256; i++){
			float result = tex1D(texture_float_1D,i);
			data3[i] = result;
		
		}
       
	}
}
