#include "module.hpp"
#include "function.hpp"
#include <stdio.h>
#include <cstring>
#include <helper_math.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
using namespace NodeCuda;

Persistent<FunctionTemplate> Module::constructor_template;
extern "C" void setTextureFilterMode();

void Module::Initialize(Handle<Object> target) {
  HandleScope scope;

  Local<FunctionTemplate> t = FunctionTemplate::New(Module::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("CudaModule"));

  // Module objects can only be created by load functions
  NODE_SET_METHOD(target, "moduleLoad", Module::Load);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "getFunction", Module::GetFunction);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "memTextureAlloc", Module::TextureAlloc);
}

Handle<Value> Module::New(const Arguments& args) {
  HandleScope scope;

  Module *pmem = new Module();
  pmem->Wrap(args.This());

  return args.This();
}

Handle<Value> Module::Load(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
  Module *pmodule = ObjectWrap::Unwrap<Module>(result);

  String::AsciiValue fname(args[0]);
  CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);
 
  result->Set(String::New("fname"), args[0]);
  result->Set(String::New("error"), Integer::New(error));
   
  return scope.Close(result);
}
Handle<Value> Module::TextureAlloc(const Arguments& args) {
  
   HandleScope scope;
   Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
   Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());
   
   /*volume binding*/
   v8::String::Utf8Value param1(args[0]->ToString());
   char *filename = *param1;
   size_t volumeSize = args[1]->Uint32Value();
   
   unsigned int width=256;
   unsigned int height=256;
   unsigned int depth=225;
    
   size_t size = width * height *depth * sizeof(unsigned char);  
   FILE *fp = fopen("/home/russa/git2/node-cuda/src/Bighead.den", "rb"); 
   void *h_data = (void *) malloc(size);
   
   size_t read = fread(h_data, 1, size, fp);
   fclose(fp);

   CUarray cu_array;
   CUDA_ARRAY3D_DESCRIPTOR desc;
   desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
   desc.NumChannels = 1;
   desc.Width = width;
   desc.Height = height;
   desc.Depth = depth;
   desc.Flags=0;
   
   CUresult error3 =cuArray3DCreate(&cu_array, &desc);
   
   CUDA_MEMCPY3D copyParam;
   memset(&copyParam, 0, sizeof(copyParam));
   copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
   copyParam.srcHost = h_data;
   copyParam.srcPitch = width * sizeof(unsigned char);
   copyParam.srcHeight = height;
   copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
   copyParam.dstArray = cu_array;
   copyParam.dstHeight=height;
   copyParam.WidthInBytes = width * sizeof(unsigned char);
   copyParam.Height = height;
   copyParam.Depth = depth;
    
   CUresult error4 = cuMemcpy3D(&copyParam);
   
   CUtexref cu_texref;
   CUresult error5 = cuModuleGetTexRef(&cu_texref, pmodule->m_module, "tex");
   CUresult error6 = cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT);
   CUresult error7 = cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_BORDER);
   CUresult error8 = cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_BORDER);
   CUresult error10 =cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR);
   CUresult error11 =cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES);
   CUresult error12 =cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_UNSIGNED_INT8, 1);
  
  
  /* Transfer Function  */
    float4 *input_float_1D = (float4 *)malloc(sizeof(float4)*256);
    for(int i=0; i<=80; i++){    //alpha
		 input_float_1D[i].x = 0.0f;
		 input_float_1D[i].y = 0.0f;
		 input_float_1D[i].z = 0.0f;
		 input_float_1D[i].w = 0.0f;
	}
	for(int i=80+1; i<=120; i++){
		input_float_1D[i].x = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].y = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].z = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		input_float_1D[i].w = (1.0f / (120.0f-80.0f)) * ( i - 80.0f);
		
	}
	for(int i=120+1; i<256; i++){
		input_float_1D[i].x =1.0f;
		input_float_1D[i].y =1.0f;
		input_float_1D[i].z =1.0f;
		input_float_1D[i].w =1.0f;
	}
	
   // Create the array on the device
   CUarray otf_array;
   CUDA_ARRAY_DESCRIPTOR ad;
   ad.Format = CU_AD_FORMAT_FLOAT;
   ad.Width = 256;
   ad.Height = 1;
   ad.NumChannels = 4;
   CUresult error13 = cuArrayCreate(&otf_array, &ad);
   
   // Copy the host input to the array
   CUresult error14 = cuMemcpyHtoA(otf_array,0,input_float_1D,256*sizeof(float4));
   
   // Texture Binding
   CUtexref otf_texref;
   CUresult error15 =cuModuleGetTexRef(&otf_texref, pmodule->m_module, "texture_float_1D");
   CUresult error16 =cuTexRefSetFilterMode(otf_texref, CU_TR_FILTER_MODE_LINEAR);
   CUresult error17 =cuTexRefSetAddressMode(otf_texref, 0, CU_TR_ADDRESS_MODE_CLAMP );
   CUresult error18 =cuTexRefSetFlags(otf_texref, CU_TRSF_NORMALIZED_COORDINATES);
   CUresult error19 =cuTexRefSetFormat(otf_texref, CU_AD_FORMAT_FLOAT, 4);
   CUresult error20 =cuTexRefSetArray(otf_texref, otf_array, CU_TRSA_OVERRIDE_FORMAT);
  
   free(h_data);
   free(input_float_1D);
  
   return scope.Close(result);
}

Handle<Value> Module::GetFunction(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = NodeCuda::Function::constructor_template->InstanceTemplate()->NewInstance();
  Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());
  NodeCuda::Function *pfunction = ObjectWrap::Unwrap<NodeCuda::Function>(result);

  String::AsciiValue name(args[0]);
  CUresult error = cuModuleGetFunction(&(pfunction->m_function), pmodule->m_module, *name);

  result->Set(String::New("name"), args[0]);
  result->Set(String::New("error"), Integer::New(error));

  return scope.Close(result);
}

