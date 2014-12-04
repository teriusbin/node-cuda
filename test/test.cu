
#include <cstdio>
#include <helper_math.h>

//texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex; 
//texture<int, 1, cudaReadModeElementType> transferTex; 


texture<float4,  1, cudaReadModeElementType> texture_float_1D;
extern "C" {
__global__ void helloWorld(unsigned int *data1, unsigned int *data2, float4 *data3, unsigned int width) {
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
		
		
		//for(float i = 0; i<=1.0f; i+=1.0f/256.0f){
			float4 result = tex1D(texture_float_1D,0.31640625);
			data3[0] = result;
		
		//}
       
	}
}
