var Buffer = require('buffer').Buffer;
var cu = require('../cuda');

var fs  = require('fs');
var sys = require('sys');
var Png = require('/usr/local/lib/node_modules/png').Png;
var Jpeg = require('/usr/local/lib/node_modules/jpeg').Jpeg;

for (var i = 0; i < cu.deviceCount; i++) {
	var cuDevice = new cu.Device(i);
	console.log("Device " + i + ":", cuDevice);
}
var cuCtx = new cu.Ctx(0, cu.Device(0));

/*global variable*/
var maxtrixBufferSize = 12;
var imageWidth=512;
var imageHeight=512;
var density = 0.05;
var brightness = 1.0;
var transferOffset = 0.0;
var transferScale = 1.0;

/*3D volume array*/ 
var d_output = cu.memAlloc(imageWidth*imageHeight*4);
/*
for (var i = 0; i < imageWidth*imageHeight; i++) {
    d_outputBuffer.writeUInt8(0, i);
}*/
//var error = d_output.copyHtoD(d_outputBuffer);

/*view vector*/ 
var c_invViewMatrix = new Buffer(12*4);
c_invViewMatrix.writeFloatLE( -1.0, 0*4);
c_invViewMatrix.writeFloatLE(  0.0, 1*4);
c_invViewMatrix.writeFloatLE(  0.0, 2*4);
c_invViewMatrix.writeFloatLE(  0.0, 3*4);
c_invViewMatrix.writeFloatLE(  0.0, 4*4);
c_invViewMatrix.writeFloatLE(  0.0, 5*4);
c_invViewMatrix.writeFloatLE( -1.0, 6*4);
c_invViewMatrix.writeFloatLE( -3.0, 7*4);
c_invViewMatrix.writeFloatLE(  0.0, 8*4);
c_invViewMatrix.writeFloatLE( -1.0, 9*4);
c_invViewMatrix.writeFloatLE(  0.0, 10*4);
c_invViewMatrix.writeFloatLE(  0.0, 11*4);

var d_invViewMatrix = cu.memAlloc(12*4);
var error = d_invViewMatrix.copyHtoD(c_invViewMatrix);

//cuModuleLoad
var cuModule = cu.moduleLoad("test.ptx");
console.log("module", cuModule);

var filename='Bighead.den';
var volumeSize=256*256*225;
var error = cuModule.memTextureAlloc("Bighead.den",256*256*225);
console.log("file read", error);

//cuModuleGetFunction
var cuFunction = cuModule.getFunction("render_kernel");

//cuLaunchKernel
var time = new Date().getTime();
var error = cu.launch(cuFunction, [32, 32, 1], [16, 16, 1],
[
	{
		type: "DevicePtr",
		value: d_output.devicePtr
	},{
		type: "DevicePtr",
		value: d_invViewMatrix.devicePtr
	},{
		type: "Uint32",
		value: imageWidth
	},{
		type: "Uint32",
		value: imageHeight
	},{
		type: "Float32",
		value: density
	},{
		type: "Float32",
		value: brightness
	},{
		type: "Float32",
		value: transferOffset
	},{
		type: "Float32",
		value: transferScale
	}
]);

console.log("Launched kernel:", error);

// cuMemcpyDtoH
var d_outputBuffer = new Buffer(imageWidth*imageHeight*4);
var error = d_output.copyDtoH(d_outputBuffer, false);
//console.log("cuda time ", (new Date().getTime() - time)/1000);
//console.log("cuda" ,d_outputBuffer);
//console.log("result", d_outputBuffer.readUInt8(0));

var png = new Png(d_outputBuffer, 512, 512, 'rgb');
var png_image = png.encodeSync();

fs.writeFileSync('./png.png', png_image.toString('binary'), 'binary');

var jpeg = new Jpeg(d_outputBuffer, 512, 512, 'rgba');
var jpeg_img = jpeg.encodeSync().toString('binary');

fs.writeFileSync('./jpeg.jpeg', jpeg_img, 'binary');

var error = cuCtx.synchronize(function(error) {
    console.log("Context synchronize with error code: " + error);

    var error = d_output.free();
   // console.log("Mem Free with error code: " + error);
    var error = d_invViewMatrix.free();
    //cuCtxDestroy
    error = cuCtx.destroy();
 
});


