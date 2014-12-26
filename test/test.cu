
#include <cstdio>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;    
texture<float4,  1, cudaReadModeElementType> texture_float_1D;


struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{

    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}


__device__
float3 mul(const float *M, const float3 &v)
{
   float3 r;
   
   r.x = v.x * M[0] + v.y * M[1] + v.z * M[2];
   r.y = v.x * M[4] + v.y * M[5] + v.z * M[6];
   r.z = v.x * M[8] + v.y * M[9] + v.z * M[10];
   
   return r;
}

__device__
float4 mul(const float *M, const float4 &v)
{
	float4 r;

	r.x = v.x * M[0] + v.y * M[1] + v.z * M[2]  + v.w * M[3];
	r.y = v.x * M[4] + v.y * M[5] + v.z * M[6]  + v.w * M[7];
	r.z = v.x * M[8] + v.y * M[9] + v.z * M[10] + v.w * M[11];	
	r.w = 1.0f;
	
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


extern "C" {
__global__ void render_kernel_volume(uint *d_output, 
								  float *d_invViewMatrix, 
								  unsigned int imageW,
								  unsigned int imageH,
								  float density,
								  float brightness,
								  float transferOffset,
								  float transferScale) 
{
	
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float opacityThreshold = 0.95f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	 
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	 
		if ((x >= imageW) || (y >= imageH)) return;
	 
		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;
	 
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);
	 
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	 
		if (!hit) return;
	 
		if (tnear < 0.0f) tnear = 0.0f; 
	 
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;
	 
		for (float i=0; i<maxSteps; i++){
				
				float sample = tex3D(tex,pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
				
				float4 col = tex1D(texture_float_1D, (sample-transferOffset)*transferScale);
     
				col.x *= col.w;
				col.y *= col.w;
				col.z *= col.w;
				
				sum = sum + col*(1.0f - sum.w);
     
				if (sum.w > opacityThreshold)
					break;
					
				t += (tstep*0.5);

				if (t > tfar) break;

				pos += (step*0.5);
  
		}
		sum.w=0.0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}
extern "C" {
__global__ void render_kernel_MIP(uint *d_output, 
								  float *d_invViewMatrix, 
								  unsigned int imageW,
								  unsigned int imageH,
								  float density,
								  float brightness,
								  float transferOffset,
								  float transferScale) 
{
	
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
	 
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	 
		if ((x >= imageW) || (y >= imageH)) return;
	 
		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;
	 
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);
	 
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	 
		if (!hit) return;
	 
		if (tnear < 0.0f) tnear = 0.0f; 
	 
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;
		float max = 0.0f; 
		for (float i=0; i<maxSteps; i++){
				
				float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
				if(sample >= max) 
					max = sample;
					
				t += (tstep*0.5);

			   if (t > tfar) break;

				pos += (step*0.5);
			
		}
		sum.x = max;
		sum.y = max;
		sum.z = max;
		sum.w = 0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}
extern "C" {
__global__ void render_kernel_MRI(uint *d_output, 
								  float *d_invViewMatrix, 
								  unsigned int imageW,
								  unsigned int imageH,
								  float density,
								  float brightness,
								  float transferOffset,
								  float transferScale) 
	{
		const int maxSteps = 500;
		const float tstep = 0.01f;
		const float opacityThreshold = 0.95f;
		const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
		const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

		uint x = blockIdx.x*blockDim.x + threadIdx.x;
		uint y = blockIdx.y*blockDim.y + threadIdx.y;

		if ((x >= imageW) || (y >= imageH)) return;

		float u = (x / (float) imageW)*2.0f-1.0f;
		float v = (y / (float) imageH)*2.0f-1.0f;

		// calculate eye ray in world space
		Ray eyeRay;
		eyeRay.o = make_float3(mul(d_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
		eyeRay.d = normalize(make_float3(u, v, -2.0f));
		eyeRay.d = mul(d_invViewMatrix, eyeRay.d);

		// find intersection with box
		float tnear, tfar;
		int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

		if (!hit) return;

		if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

		// march along ray from front to back, accumulating color
		float4 sum = make_float4(0.0f);
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d * tnear;
		float3 step = eyeRay.d*tstep;
		
		float max = 0.0f; 
		
				
		float sample = tex3D(tex, pos.x+0.5f, pos.y+0.5f+transferOffset, pos.z+0.5f);
				
		sum.x = sample;
		sum.y = sample;
		sum.z = sample;
		sum.w = 0;
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
	}
}
