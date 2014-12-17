#include <cstring>
#include <node_buffer.h>
#include "mem.hpp"
#include "module.hpp"
#include <stdio.h>

using namespace NodeCuda;

typedef struct bitmapFileHeader{
	unsigned short int bfType;  
	unsigned int bfSize; 
	unsigned short int bfReserved1, bfReserved2;
	unsigned int bfOffBits;
};
typedef struct bitmapInfoHeader{
	unsigned int biSize;
	int biWidth, biHeight;
	unsigned short int biPlanes; 
	unsigned short int biBitCount;
	unsigned int biCompression; 
	unsigned int biSizeImage;
	int biXPelsPerMeter, biYPelsPerMeter;
	unsigned int biClrUsed;
	unsigned int biClrImportant; 
};

Persistent<FunctionTemplate> Mem::constructor_template;

void Mem::Initialize(Handle<Object> target) {
  HandleScope scope;

  Local<FunctionTemplate> t = FunctionTemplate::New(Mem::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("CudaMem"));

  // Mem objects can only be created by allocation functions
  NODE_SET_METHOD(target, "memAlloc", Mem::Alloc);
  NODE_SET_METHOD(target, "memAllocPitch", Mem::AllocPitch);
 
  constructor_template->InstanceTemplate()->SetAccessor(String::New("devicePtr"), Mem::GetDevicePtr);

  NODE_SET_PROTOTYPE_METHOD(constructor_template, "free", Mem::Free);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "copyHtoD", Mem::CopyHtoD);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "copyDtoH", Mem::CopyDtoH);
}

Handle<Value> Mem::New(const Arguments& args) {
  HandleScope scope;

  Mem *pmem = new Mem();
  pmem->Wrap(args.This());

  return args.This();
}


Handle<Value> Mem::Alloc(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
  Mem *pmem = ObjectWrap::Unwrap<Mem>(result);

  size_t bytesize = args[0]->Uint32Value();
  CUresult error = cuMemAlloc(&(pmem->m_devicePtr), bytesize);
  printf("memalloc %d\n",bytesize);
  
  result->Set(String::New("size"), Integer::NewFromUnsigned(bytesize));
  result->Set(String::New("error"), Integer::New(error));

  return scope.Close(result);
}

Handle<Value> Mem::AllocPitch(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
  Mem *pmem = ObjectWrap::Unwrap<Mem>(result);

  size_t pPitch;
  unsigned int ElementSizeBytes = args[2]->Uint32Value();
  size_t WidthInBytes = ElementSizeBytes * args[0]->Uint32Value();
  size_t Height = args[1]->Uint32Value();
  CUresult error = cuMemAllocPitch(&(pmem->m_devicePtr), &pPitch, WidthInBytes, Height, ElementSizeBytes);

  result->Set(String::New("size"), Integer::NewFromUnsigned(pPitch * Height));
  result->Set(String::New("pitch"), Integer::NewFromUnsigned(pPitch));
  result->Set(String::New("error"), Integer::New(error));

  return scope.Close(result);
}

Handle<Value> Mem::Free(const Arguments& args) {
  HandleScope scope;
  Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());

  CUresult error = cuMemFree(pmem->m_devicePtr);

  return scope.Close(Number::New(error));
}

Handle<Value> Mem::CopyHtoD(const Arguments& args) {
  HandleScope scope;
  Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());

  Local<Object> buf = args[0]->ToObject();
  char *phost = Buffer::Data(buf);
  size_t bytes = Buffer::Length(buf);

  bool async = args.Length() >= 2 && args[1]->IsTrue();
  printf("CopyHtoD %d\n",bytes);
  CUresult error;
  if (async) {
    error = cuMemcpyHtoDAsync(pmem->m_devicePtr, phost, bytes, 0);
  } else {
    error = cuMemcpyHtoD(pmem->m_devicePtr, phost, bytes);
  }

  return scope.Close(Number::New(error));
}
void ScreenCapture( const char *strFilePath , char *d_output)
{
 
    bitmapFileHeader BMFH;
    bitmapInfoHeader BMIH;
    
    int nWidth = 0;
    int nHeight = 0;
    unsigned long dwQuadrupleWidth = 0;     //LJH Ãß°¡, °¡·Î »çÀÌÁî°¡ 4ÀÇ ¹èŒö°¡ ŸÆŽÏ¶óžé 4ÀÇ ¹èŒö·Î žžµéŸîŒ­ ÀúÀå
 
    //GLbyte *pPixelData = NULL;              //front bufferÀÇ ÇÈŒ¿ °ªµéÀ» ŸòŸî ¿À±â À§ÇÑ ¹öÆÛÀÇ Æ÷ÀÎÅÍ
 
    nWidth  = 512;     //(³ªÀÇ °æ¿ì)ž®Žªœº¿¡Œ­ÀÇ °æ¿ì ÇØ»óµµ °íÁ€ÀÌ¹Ç·Î ±× °ªÀ» ÀÔ·Â
    nHeight = 512;
 
 
    dwQuadrupleWidth = ( nWidth % 4 ) ? ( ( nWidth ) + ( 4 - ( nWidth % 4 ) ) ) : ( nWidth );
 
    //ºñÆ®žÊ ÆÄÀÏ ÇìŽõ Ã³ž®
    BMFH.bfType=0x4D42;      //B(42)¿Í M(4D)¿¡ ÇØŽçÇÏŽÂ ASCII °ªÀ» ³ÖŸîÁØŽÙ.
    //¹ÙÀÌÆ® ŽÜÀ§·Î ÀüÃŒÆÄÀÏ Å©±â
    BMFH.bfSize=sizeof( bitmapFileHeader ) + sizeof( bitmapInfoHeader ) + ( dwQuadrupleWidth * 3 * nHeight );
    //¿µ»ó µ¥ÀÌÅÍ À§Ä¡±îÁöÀÇ °Åž®
    BMFH.bfOffBits=sizeof( bitmapFileHeader ) + sizeof( bitmapInfoHeader );
 
    //ºñÆ®žÊ ÀÎÆ÷ ÇìŽõ Ã³ž®
    BMIH.biSize=sizeof( bitmapInfoHeader );       //ÀÌ ±žÁ¶ÃŒÀÇ Å©±â
    BMIH.biWidth=nWidth;                           //ÇÈŒ¿ ŽÜÀ§·Î ¿µ»óÀÇ Æø
    BMIH.biHeight           = nHeight;                          //¿µ»óÀÇ ³ôÀÌ
    BMIH.biPlanes           = 1;                                //ºñÆ® ÇÃ·¹ÀÎ Œö(Ç×»ó 1)
    BMIH.biBitCount        = 24;                               //ÇÈŒ¿Žç ºñÆ®Œö(ÄÃ·¯, Èæ¹é ±žº°)
    BMIH.biCompression     = 0;                           //ŸÐÃà À¯¹«
    BMIH.biSizeImage        = 512 * 3 * 512;					//¿µ»óÀÇ Å©±â
    BMIH.biXPelsPerMeter   = 0;                                //°¡·Î ÇØ»óµµ
    BMIH.biYPelsPerMeter    = 0;                                //ŒŒ·Î ÇØ»óµµ
    BMIH.biClrUsed        = 0;                                //œÇÁŠ »ç¿ë »ö»óŒö
    BMIH.biClrImportant    = 0;                                //Áß¿äÇÑ »ö»ó ÀÎµŠœº
 
    //pPixelData = new GLbyte[ dwQuadrupleWidth * 3 * nHeight ];  //LJH ŒöÁ€
 
    //ÇÁ·±Æ® ¹öÆÛ·Î ºÎÅÍ ÇÈŒ¿ Á€ºžµéÀ» ŸòŸî¿ÂŽÙ.
    //glReadPixels(
    //    0, 0,                   //ÄžÃ³ÇÒ ¿µ¿ªÀÇ ÁÂÃø»óŽÜ ÁÂÇ¥
    //    nWidth, nHeight,        //ÄžÃ³ÇÒ ¿µ¿ªÀÇ Å©±â
    //    GL_BGR,                 //ÄžÃ³ÇÑ ÀÌ¹ÌÁöÀÇ ÇÈŒ¿ Æ÷žË
    //    GL_UNSIGNED_BYTE,       //ÄžÃ³ÇÑ ÀÌ¹ÌÁöÀÇ µ¥ÀÌÅÍ Æ÷žË
    //    pPixelData              //ÄžÃ³ÇÑ ÀÌ¹ÌÁöÀÇ Á€ºžžŠ ŽãŸÆµÑ ¹öÆÛ Æ÷ÀÎÅÍ
    //    );
 
    {//ÀúÀå ºÎºÐ
        FILE *outFile = fopen( strFilePath, "wb" );
        if( outFile == NULL )
        {
            //¿¡·¯ Ã³ž®
            //printf( "¿¡·¯" );
            //fclose( outFile );
        }
        fwrite( &BMFH, sizeof( unsigned char ), sizeof(bitmapFileHeader), outFile );         //ÆÄÀÏ ÇìŽõ Ÿ²±â
        fwrite( &BMIH, sizeof( unsigned char ), sizeof(bitmapInfoHeader), outFile );         //ÀÎÆ÷ ÇìŽõ Ÿ²±â
		for(int i=0; i<512; i++){
			for(int j=0; j<512; j++){
				fwrite((d_output+(i*512 +j+0)), sizeof( unsigned char ), 1, outFile );
				fwrite((d_output+(i*512 +j+1)), sizeof( unsigned char ), 1, outFile );
				fwrite((d_output+(i*512 +j+2)), sizeof( unsigned char ), 1, outFile );
			}
		}
        //fwrite( d_output, sizeof( uchar ), BMIH.biSizeImage, outFile );   //c_outputÆÄÀÏ·Î ÀÐÀº µ¥ÀÌÅÍ Ÿ²±â
 
        fclose( outFile );  //ÆÄÀÏ ŽÝ±â
    }
 
    if ( d_output != NULL )
    {
        delete d_output;
    }
}
Handle<Value> Mem::CopyDtoH(const Arguments& args) {
  HandleScope scope;
  Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());

  Local<Object> buf = args[0]->ToObject();
  char *phost = Buffer::Data(buf);
  size_t bytes = Buffer::Length(buf);
  
  printf("CopyDtoH %d\n",bytes);
  bool async = args.Length() >= 2 && args[1]->IsTrue();
  CUresult error;
  if (async) {
    error = cuMemcpyDtoHAsync(phost, pmem->m_devicePtr, bytes, 0);
  } else {
    error = cuMemcpyDtoH(phost, pmem->m_devicePtr, bytes);
    //const char* szStr = "dldndrb1";
	//ScreenCapture(szStr,phost);
  }
  
  return scope.Close(Number::New(error));
}

Handle<Value> Mem::GetDevicePtr(Local<String> property, const AccessorInfo &info) {
  HandleScope scope;
  Mem *pmem = ObjectWrap::Unwrap<Mem>(info.Holder());
  Buffer *ptrbuf = Buffer::New(sizeof(pmem->m_devicePtr));

  memcpy(Buffer::Data(ptrbuf->handle_), &pmem->m_devicePtr, sizeof(pmem->m_devicePtr));

  return scope.Close(ptrbuf->handle_);
}
