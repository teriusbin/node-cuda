
#include <cstdio>
#include <helper_math.h>


texture<float4,  1, cudaReadModeElementType> texture_float_1D;
texture<float, 3, cudaReadModeElementType> tex;
extern "C" {
__global__ void helloWorld(float *data3, unsigned int width) {
	    
	    /*matrix mul sample*/
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
		
		 /*1D OTF Table sample*/
		/*
		for(float i = 0; i<=1.0f; i+=1.0f/256.0f){
			float4 result = tex1D(texture_float_1D,0.31640625);
			data3[0] = result;
		
		}
		*/
		
		
		/*3D array sample*/
		int loop;
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int z;
	  
		for(loop=0; loop<2*2*2; loop++){
			z = loop;
			data3[z*2*2 + y*2 + x] = tex3D(tex, x, y, z);
		}
		
	}
}
