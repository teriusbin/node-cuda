// Test
var Buffer = require('buffer').Buffer;
var cu = require('../cuda');

var cuCtx = new cu.Ctx(0, cu.Device(0));

// ~ Mem alloac and copy start




// ~ cumem1
var cuMem1 = cu.memAlloc(65536*4);
//console.log("cuMem Allocated 65536*4 bytes:", cuMem1);

var buf1 = new Buffer(65536*4);
for (var i = 0; i < 65536; i++) {
    buf1.writeInt32LE(1, i*4);
}
var error = cuMem1.copyHtoD(buf1);
//console.log("Buf:", buf1);
//console.log("Copied buffer to device:!!!!", error);
//console.log("-------------------------------------------------------------");


// ~ cumem2
var cuMem2 = cu.memAlloc(65536*4);
//console.log("cuMem Allocated 65536*4 bytes:", cuMem2);

var buf2 = new Buffer(65536*4);
for (var i = 0; i < 65536; i++) {
    buf2.writeInt32LE(1, i*4);
}
var error = cuMem2.copyHtoD(buf2);
//console.log("Buf:", buf2);
//console.log("Copied buffer to device!!!:", error);
//console.log("-------------------------------------------------------------");

// ~ cumem3
var cuMem3 = cu.memAlloc(256*4);
//console.log("cuMem Allocated 65536*4 bytes:", cuMem3);

var buf3 = new Buffer(256*4);
for (var i = 0; i < 256; i++) {
    buf3.writeInt32LE(0, i*4);
}
var error = cuMem3.copyHtoD(buf3);
//console.log("Copied buffer to device:!!!!", error);
//console.log("-------------------------------------------------------------");

//cuModuleLoad
var cuModule = cu.moduleLoad("test.ptx");
console.log("module", cuModule);


var filename='Bighead.den';
var volumeSize=256*256*225;
var error = cuModule.memTextureAlloc(filename, volumeSize);
console.log("file read", error);


//cuModuleGetFunction
var cuFunction = cuModule.getFunction("helloWorld");

//cuLaunchKernel
var imageWidth=256;
var time = new Date().getTime();
var error = cu.launch(cuFunction, [1, 1, 1], [1, 1, 1],
[
	{
		type: "DevicePtr",
		value: cuMem1.devicePtr
	}, {
		type: "DevicePtr",
		value: cuMem2.devicePtr
	},{
		type: "DevicePtr",
		value: cuMem3.devicePtr
	},{
		type: "Uint32",
		value: imageWidth
	}
]);

console.log("Launched kernel:", error);

// cuMemcpyDtoH
var error = cuMem3.copyDtoH(buf3, true);
//console.log("cuda time ", (new Date().getTime() - time)/1000);
console.log("cuda" ,buf3);
console.log("float", buf3.readFloatLE(119*4));

console.log('----------------------------------------');
//console.log("--------------------------------cuda buffer");


//cuCtxSynchronize
var error = cuCtx.synchronize(function(error) {
    console.log("Context synchronize with error code: " + error);

    //cuMemFree
    var error = cuMem1.free();
   // console.log("Mem Free with error code: " + error);
    
    var error = cuMem2.free();
    //console.log("Mem Free with error code: " + error);
    
    var error = cuMem3.free();
   // console.log("Mem Free with error code: " + error);

    //cuCtxDestroy
    error = cuCtx.destroy();
   // console.log("Context destroyed with error code: " + error);
});
