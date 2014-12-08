var Buffer = require('buffer').Buffer;
var cu = require('../cuda');


for (var i = 0; i < cu.deviceCount; i++) {
	var cuDevice = new cu.Device(i);
	console.log("Device " + i + ":", cuDevice);
}
var cuCtx = new cu.Ctx(0, cu.Device(0));

/*float4 OTF table*/ 
/*
var cuMem3 = cu.memAlloc(256*4*4);
var buf3 = new Buffer(256*4*4);
for (var i = 0; i < 256*4; i++) {
	buf3.writeInt32LE(0, i*4);
}
var error = cuMem3.copyHtoD(buf3);
*/

/*3D float array*/ 
/*
var cuMem3 = cu.memAlloc(2*2*2*4);

var buf3 = new Buffer(2*2*2*4);
for (var i = 0; i < 2*2*2; i++) {
    buf3.writeFloatLE(0, i*4);
}
var error = cuMem3.copyHtoD(buf3);
*/

/*3D volume array*/ 
var cuMem3 = cu.memAlloc(256*256*225*1);

var buf3 = new Buffer(256*256*225*1);
for (var i = 0; i < 256*256*225; i++) {
    buf3.writeUInt8(0, i*1);
}
var error = cuMem3.copyHtoD(buf3);

//cuModuleLoad
var cuModule = cu.moduleLoad("test.ptx");
console.log("module", cuModule);


var filename='Bighead.den';
var volumeSize=256*256*225;
var error = cuModule.memTextureAlloc("Bighead.den",256*256*225);
console.log("file read", error);


//cuModuleGetFunction
var cuFunction = cuModule.getFunction("helloWorld");

//cuLaunchKernel
var imageWidth=256;
var time = new Date().getTime();
var error = cu.launch(cuFunction, [32, 32, 1], [8, 8, 8],
[
	{
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

/*float 3d integer*/ 
//console.log("float", buf3.readFloatLE(3*4));

/*float4 OTF table*/ 
/*
console.log("float", buf3.readFloatLE(0*4*4));
*/

/*uint 3d volume*/ 
console.log("uint", buf3.readUInt8(108511));
console.log('----------------------------------------');

var error = cuCtx.synchronize(function(error) {
    console.log("Context synchronize with error code: " + error);

    var error = cuMem3.free();
   // console.log("Mem Free with error code: " + error);

    //cuCtxDestroy
    error = cuCtx.destroy();
 
});
