import tensorrt as trt
print(trt.__version__)
import numpy as np
import os

import pycuda.autoinit
import pycuda.driver as cuda

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self,engine_path,dtype=np.float32,max_batch_size=1,model_height=1,model_width=1):
        self.cfx = cuda.Device(0).make_context()
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.input_shape, self.output_shape, self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        input_shape, output_shape = [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * self.max_batch_size
            shape = trt.Dims(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem_input = cuda.pagelocked_empty(size, dtype)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem_input, device_mem))
                input_shape = shape
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_shape.append(shape)
        return input_shape, output_shape, inputs, outputs, bindings, stream

    def __call__(self,x,batch_size=1):
        self.cfx.push()
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        self.cfx.pop()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]