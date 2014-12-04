#include "module.hpp"
#include "function.hpp"
#include <stdio.h>
#include <cstring>
#include <helper_math.h>
#include <stdlib.h>
using namespace NodeCuda;

Persistent<FunctionTemplate> Module::constructor_template;

void Module::Initialize(Handle<Object> target) {
  HandleScope scope;

  Local<FunctionTemplate> t = FunctionTemplate::New(Module::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("CudaModule"));

  // Module objects can only be created by load functions
  NODE_SET_METHOD(target, "moduleLoad", Module::Load);
  //NODE_SET_METHOD(target, "memTextureAlloc", Module::TextureAlloc);
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
   //result->Set(String::New("error!!!!!!!!!!!!!!!!"), Integer::New(error4));
  return scope.Close(result);
}
Handle<Value> Module::TextureAlloc(const Arguments& args) {
  
   HandleScope scope;
   Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
   Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());
   
   v8::String::Utf8Value param1(args[0]->ToString());
   char *filename = *param1;
   size_t volumeSize = args[1]->Uint32Value();
   /*
   FILE *fp = fopen("/home/russa/git2/node-cuda/src/Bighead.den", "rb");

   void *data = malloc(volumeSize);
   size_t read = fread(data, 1, volumeSize, fp);
   fclose(fp);

   printf("\n~~~~Read '%s', %d bytes\n", filename, read);
	
   CUarray cu_array;
   CUDA_ARRAY3D_DESCRIPTOR desc;
   desc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
   desc.NumChannels = 1;
   desc.Width = 256;
   desc.Height = 256;
   desc.Depth = 225;
   desc.Flags = 1;
 
   CUresult error2 = cuArray3DCreate(&cu_array, &desc);
    
   CUDA_MEMCPY3D copyParam;
   memset(&copyParam, 0, sizeof(copyParam));
   copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
   copyParam.dstArray = cu_array;
   copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
   copyParam.srcHost = data;
   copyParam.srcPitch = 256 * sizeof(unsigned char);
   copyParam.WidthInBytes = copyParam.srcPitch;
   copyParam.Height = 256;
   copyParam.Depth = 225;
   CUresult error3 =  cuMemcpy3D(&copyParam);

   CUtexref cu_texref;
   CUresult error4 = cuModuleGetTexRef(&cu_texref, pmodule->m_module , "tex");
   CUresult error5 = cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT);
   CUresult error6 = cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_CLAMP );
   CUresult error7 = cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_CLAMP );
   CUresult error8 = cuTexRefSetAddressMode(cu_texref, 2, CU_TR_ADDRESS_MODE_CLAMP );
   CUresult error9 = cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR);
   CUresult error10 = cuTexRefSetFlags(cu_texref, CU_TRSF_READ_AS_INTEGER);
   CUresult error11 = cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_UNSIGNED_INT32, 1);
   
   result->Set(String::New("volumeSize"), Integer::NewFromUnsigned(volumeSize));
   result->Set(String::New("cuArray3DCreate error"), Integer::New(error2));
   result->Set(String::New("cuMemcpy3D error"), Integer::New(error3));
   result->Set(String::New("cuModuleGetTexRef error"), Integer::New(error4));
   result->Set(String::New("cuTexRefSetArray error"), Integer::New(error5));
   result->Set(String::New("cuTexRefSetAddressMode error"), Integer::New(error6));
   result->Set(String::New("cuTexRefSetAddressMode error"), Integer::New(error7));
   result->Set(String::New("cuTexRefSetAddressMode error"), Integer::New(error8));
   result->Set(String::New("cuTexRefSetFilterMode error"), Integer::New(error9));
   result->Set(String::New("cuTexRefSetFlags error"), Integer::New(error10));
   result->Set(String::New("cuTexRefSetFormat error"), Integer::New(error11));
   */
   /*
   float *transferFunc = (float *)malloc(sizeof(float)*256);
  
	 for(int i=0; i<=80; i++){    //alpha
		 transferFunc[i] = 1.0f;
		 //transferFunc[i].x = 1.0f;
		 //transferFunc[i].y = 1.0f;
		 //transferFunc[i].z = 1.0f;
	}
	for(int i=80+1; i<=120; i++){
		transferFunc[i] = (1.0 / (120-80)) * ( i - 80);
		//transferFunc[i].x = (1.0 / (120-80)) * ( i - 80);
		//transferFunc[i].y = (1.0 / (120-80)) * ( i - 80);
		//transferFunc[i].z = (1.0 / (120-80)) * ( i - 80);
	}
	for(int i=120+1; i<256; i++){
		transferFunc[i] =1.0f;
		//transferFunc[i].x =1.0f;
		//transferFunc[i].y =1.0f;
		//transferFunc[i].z =1.0f;
	}
   
   for(int i=0; i<256; i++){
	   printf("%d    %f \n",i, transferFunc[i]);
   }
   
   
   CUarray cu_transfer;
   CUDA_ARRAY_DESCRIPTOR desc2;
   desc2.Format = CU_AD_FORMAT_FLOAT;
   desc2.Width = 256;
   desc2.Height = 1;
   desc2.NumChannels = 4;
   CUresult error12 = cuArrayCreate(&cu_transfer, &desc2);
   
   
   cuMemcpyHtoA(cu_transfer,0,transferFunc,256*sizeof(float));
   /*
   CUDA_MEMCPY2D copyParam2;
   memset(&copyParam2, 0, sizeof(copyParam2));
   copyParam2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
   copyParam2.dstArray = cu_transfer;
   copyParam2.srcMemoryType = CU_MEMORYTYPE_HOST;
   copyParam2.srcHost = transferFunc;
   copyParam2.srcPitch = 256 * sizeof(float);
   copyParam2.WidthInBytes = copyParam2.srcPitch;
   copyParam2.Height = 1;
   
   CUresult error13 =  cuMemcpy2D(&copyParam2);
  
   CUtexref cu_transfer_tex;
   CUresult error14 =cuModuleGetTexRef(&cu_transfer_tex, pmodule->m_module, "transferTex");
   CUresult error17 =cuTexRefSetFilterMode(cu_transfer_tex, CU_TR_FILTER_MODE_LINEAR);
   CUresult error16 =cuTexRefSetAddressMode(cu_transfer_tex, 0, CU_TR_ADDRESS_MODE_CLAMP);
   CUresult error18 =cuTexRefSetFlags(cu_transfer_tex, CU_TRSF_NORMALIZED_COORDINATES);
   CUresult error19 =cuTexRefSetFormat(cu_transfer_tex, CU_AD_FORMAT_FLOAT, 1);
   CUresult error15 =cuTexRefSetArray(cu_transfer_tex, cu_transfer, CU_TRSA_OVERRIDE_FORMAT);
   

    //cuParamSetTexRef(function, CU_PARAM_TR_DEFAULT, cu_transfer_tex);
    
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuArrayCreate error"), Integer::New(error12));
   //result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuMemcpy2D error"), Integer::New(error13));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuModuleGetTexRef error"), Integer::New(error14));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuTexRefSetArray error"), Integer::New(error15));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuTexRefSetAddressMode error"), Integer::New(error16));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuTexRefSetFilterMode error"), Integer::New(error17));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuTexRefSetFlags error"), Integer::New(error18));
   result->Set(String::New("~~~~~~~~~~~~~~~~~~~cuTexRefSetFormat error"), Integer::New(error19));
   
   */
  
   //free(data);
   //free(transferFunc);
   
  /* float  array 256  */
  
    float4 *input_float_1D = (float4 *)malloc(sizeof(float4)*256);
    for(int i=0; i<=80; i++){    //alpha
		 input_float_1D[i].x = 1.0f;
		 input_float_1D[i].y = 1.0f;
		 input_float_1D[i].z = 1.0f;
		 input_float_1D[i].w = 1.0f;
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
	for(int i=0; i<256; i++){
	   printf("%d  %f %f  %f %f \n",i, input_float_1D[i].x,input_float_1D[i].y,input_float_1D[i].z,input_float_1D[i].w);
   }
   
     // Create the array on the device
   CUarray array;
   CUDA_ARRAY_DESCRIPTOR ad;
   ad.Format = CU_AD_FORMAT_FLOAT;
   ad.Width = 256;
   ad.Height = 1;
   ad.NumChannels = 4;
   CUresult error12 = cuArrayCreate(&array, &ad);
   
   // Copy the host input to the array
   cuMemcpyHtoA(array,0,input_float_1D,256*sizeof(float4));
   
   CUtexref texref;
   CUresult error14 =cuModuleGetTexRef(&texref, pmodule->m_module, "texture_float_1D");
   CUresult error17 =cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_POINT );
   CUresult error16 =cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP );
   CUresult error18 =cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES);
   CUresult error19 =cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 4);
   CUresult error15 =cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);
   
   
    /* integer  array 256  */
    /*
    int *input_float_1D = (int *)malloc(sizeof(int)*256);
	for(int i=0; i<256; i++){
		input_float_1D[i] =i;
		
	}
	for(int i=0; i<256; i++){
	   printf("%d    %d \n",i, input_float_1D[i]);
   }
   // Create the array on the device
   CUarray array;
   CUDA_ARRAY_DESCRIPTOR ad;
   ad.Format = CU_AD_FORMAT_UNSIGNED_INT32;
   ad.Width = 256;
   ad.Height = 1;
   ad.NumChannels = 1;
   CUresult error12 = cuArrayCreate(&array, &ad);
   
   // Copy the host input to the array
   cuMemcpyHtoA(array,0,input_float_1D,256*sizeof(int));
   
   CUtexref texref;
   CUresult error14 =cuModuleGetTexRef(&texref, pmodule->m_module, "texture_float_1D");
   CUresult error17 =cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_POINT );
   CUresult error16 =cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP );
   CUresult error18 =cuTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER);
   CUresult error19 =cuTexRefSetFormat(texref, CU_AD_FORMAT_UNSIGNED_INT32, 1);
   CUresult error15 =cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);
  */
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

