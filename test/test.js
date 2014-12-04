var Buffer = require('buffer').Buffer;
var cu = require('../cuda');

//cuDriverGetVersion
//cuDeviceGetCount

//console.log("Node-cuda exports:", cu);
/*
for (var i = 0; i < cu.deviceCount; i++) {

    var cuDevice = new cu.Device(i);
    console.log("Device " + i + ":", cuDevice);
}
*/
//cuCtxCreate
var cuCtx = new cu.Ctx(0, cu.Device(0));

//cuCtxGetApiVersion
//console.log("Created context:", cuCtx);


//cuMemAlloc
var cuMem = cu.memAlloc(65536*4);
console.log("cuMem Allocated 144 bytes:", cuMem);

var buf = new Buffer(65536*4);
for (var i = 0; i < 65536; i++) {
    buf.writeInt32BE(1, i*4);
}
var error = cuMem.copyHtoD(buf);
console.log("Copied buffer to device:", error);
console.log("-------------------------------------------------------------");



var bufjs = new Buffer(65536);
for (var i = 0; i < bufjs.length; i++) {
    bufjs[i] = 1;
}
//console.log("Host Created buffer of 100 bytes:", buf);


var cuMem2 = cu.memAlloc(65536);
console.log("cuMem2 Allocated 144 bytes:", cuMem2);
var buf2 = new Buffer(65536);
for (var i = 0; i < buf2.length; i++) {
    buf2[i] = 1;
}

var buf2js= new Buffer(65536);
for (var i = 0; i < buf2js.length; i++) {
    buf2js[i] = 1;
}
//console.log("Host Created buffer of 144 bytes:", buf2);
var error = cuMem2.copyHtoD(buf2);
console.log("Copied buffer to device:", error);

//console.log("-------------------------------------------------------------");

var buf3js= new Buffer(65536);
for (var i = 0; i < buf3js.length; i++) {
    buf3js[i] = 0;
}
var cuMem3 = cu.memAlloc(65536);
console.log("cuMem3 Allocated 144 bytes:", cuMem3);

//console.log("-------------------------------------------------------------");

//cuModuleLoad
var cuModule = cu.moduleLoad("test.ptx");
//console.log("Loaded module:", cuModule);


//cuModuleGetFunction
var cuFunction = cuModule.getFunction("helloWorld");
//console.log("Got function:", cuFunction);

//console.log("-------------------------------------------------------------");

//cuLaunchKernel
var time = new Date().getTime();
var error = cu.launch(cuFunction, [8, 8, 1], [8, 8, 1],
[
	{
		type: "DevicePtr",
		value: cuMem.devicePtr
	}, {
		type: "DevicePtr",
		value: cuMem2.devicePtr
	},{
		type: "DevicePtr",
		value: cuMem3.devicePtr
	}
]);

console.log("Launched kernel:", error);
//console.log("--------------------------------");
console.log("~~~~~~~~~~~~~~~~~~~~~~~Copied buffer to host:", error);
// cuMemcpyDtoH
var error = cuMem3.copyDtoH(buf, true);
console.log("time ", (new Date().getTime() - time)/1000);

console.log("--------------------------------cuda buffer");
console.log(buf[1]);

var temp = function(buf1, buf2, buf3){
	
	for( var i = 0; i < 256; i++){
		for( var j = 0; j < 256; j++){
			for( var index = 0; index < 256; index++)
			{
				buf3[i*256+j] = buf3[i*256+j] + (buf2[i*256+index] * buf1[index*256+j]);
				//console.log(bufjs[i*12+index], buf2js[index*12+j], buf3js[i*12+j]);
			}
		}
	}
	
};

var time2 = new Date().getTime();
	/*for( var i = 0; i < 256; i++){
		for( var j = 0; j < 256; j++){
			for( var index = 0; index < 256; index++)
			{
				buf3js[i*256+j] = buf3js[i*256+j] + (bufjs[i*256+index] * buf2js[index*256+j]);
				//console.log(bufjs[i*12+index], buf2js[index*12+j], buf3js[i*12+j]);
			}
		}
	}
	*/
	
temp(bufjs, buf2js, buf3js);
	
console.log("time2 ", (new Date().getTime() - time2)/1000);
console.log("--------------------------------js buffer");
console.log(buf3js);


//console.log("Copied buffer to host:", error);
/*
var error = cuMem.copyDtoH(buf2, true);
console.log("Device to host", buf2);
console.log("Copied buffer to host:", error);

var error = cuMem.copyDtoH(buf3, true);
console.log("Device to host", buf3);
console.log("Copied buffer to host:", error);
*/




//cuCtxSynchronize
var error = cuCtx.synchronize(function(error) {
    console.log("Context synchronize with error code: " + error);

    //cuMemFree
    var error = cuMem.free();
    console.log("Mem Free with error code: " + error);
    
    var error = cuMem2.free();
    //console.log("Mem Free with error code: " + error);
    
    var error = cuMem3.free();
   // console.log("Mem Free with error code: " + error);

    //cuCtxDestroy
    error = cuCtx.destroy();
   // console.log("Context destroyed with error code: " + error);
});
