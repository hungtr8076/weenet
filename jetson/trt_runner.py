#!/usr/bin/env python3
"""TensorRT 8.2 runners for the Jetson Nano (JetPack 4.6.1: Python 3.6, no pycuda; the CUDA runtime is called through
ctypes). No f-strings: the board runs Python 3.6.
  Runner        synchronous: copy in, execute_v2, copy out (jetson/engine_accuracy.py)
  AsyncRunner   one CUDA stream and page-locked host buffers; enqueue() then wait() (jetson/streams.py,
                jetson/verify_batch.py)"""
from __future__ import print_function
import ctypes
import os

import numpy as np
import tensorrt as trt

cudart = ctypes.CDLL(os.environ.get("CUDART", "/usr/local/cuda/lib64/libcudart.so"))
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
H2D, D2H = 1, 2
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ck(err, what):
    if err != 0:
        raise SystemExit("CUDA error %d in %s" % (err, what))


def cuda_init():
    ck(cudart.cudaFree(ctypes.c_void_p(0)), "init")


def normalize_hwc_uint8(img):
    """uint8 HxWx3 RGB -> float32 1x3xHxW, identical maths to ToTensor + Normalize."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class Runner(object):
    def __init__(self, plan_bytes, logger):
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan_bytes)
        if self.engine is None:
            raise SystemExit("cannot deserialize engine")
        self.ctx = self.engine.create_execution_context()
        self.bufs, self.host = [], {}
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dt = trt.nptype(self.engine.get_binding_dtype(i))
            n = int(np.prod(shape)) * np.dtype(dt).itemsize
            p = ctypes.c_void_p()
            ck(cudart.cudaMalloc(ctypes.byref(p), n), "malloc")
            self.bufs.append(p)
            self.host[self.engine.get_binding_name(i)] = (i, shape, dt, n)
        if len(self.host) != 2:
            raise SystemExit("expected 1 input + 1 output binding, got %s" % list(self.host))
        self.inp = [k for k in self.host if self.engine.binding_is_input(self.host[k][0])][0]
        self.out = [k for k in self.host if k != self.inp][0]
        self.ptrs = [int(b.value) for b in self.bufs]

    def __call__(self, x):
        i, shape, dt, n = self.host[self.inp]
        xin = np.ascontiguousarray(x.reshape(shape).astype(dt))
        ck(cudart.cudaMemcpy(self.bufs[i], xin.ctypes.data, n, H2D), "h2d")
        if not self.ctx.execute_v2(self.ptrs):
            raise SystemExit("execute_v2 failed")
        j, oshape, odt, on = self.host[self.out]
        out = np.empty(oshape, dtype=odt)
        ck(cudart.cudaMemcpy(out.ctypes.data, self.bufs[j], on, D2H), "d2h")
        return out.reshape(-1)


class AsyncRunner(object):
    """One engine with one input and one output binding, one in-order CUDA stream, page-locked host buffers. Fill
    host_in() in place, call enqueue() (H2D, execute_async_v2, D2H, event), then wait() for the output."""

    def __init__(self, plan_bytes, logger):
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan_bytes)
        if self.engine is None:
            raise SystemExit("cannot deserialize engine")
        self.ctx = self.engine.create_execution_context()
        if self.engine.num_bindings != 2:
            raise SystemExit("expected 1 input + 1 output binding, got %d" % self.engine.num_bindings)
        self.i_in = [i for i in range(2) if self.engine.binding_is_input(i)][0]
        self.i_out = 1 - self.i_in
        self.meta = {}                       # binding -> (shape, dtype, bytes)
        self.dev, self.host = {}, {}
        for i in (0, 1):
            shape = tuple(self.engine.get_binding_shape(i))
            dt = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            n = int(np.prod(shape)) * dt.itemsize
            self.meta[i] = (shape, dt, n)
            d, h = ctypes.c_void_p(), ctypes.c_void_p()
            ck(cudart.cudaMalloc(ctypes.byref(d), n), "malloc")
            ck(cudart.cudaHostAlloc(ctypes.byref(h), n, 0), "hostalloc")
            self.dev[i] = d
            self.host[i] = (h, np.frombuffer((ctypes.c_char * n).from_address(h.value), dtype=dt).reshape(shape))
        self.bindings = [int(self.dev[0].value), int(self.dev[1].value)]
        self.stream, self.event = ctypes.c_void_p(), ctypes.c_void_p()
        ck(cudart.cudaStreamCreate(ctypes.byref(self.stream)), "stream")
        ck(cudart.cudaEventCreate(ctypes.byref(self.event)), "event")

    def host_in(self):
        """Page-locked input array (fill it in place, then call enqueue())."""
        return self.host[self.i_in][1]

    def enqueue(self):
        n_in, n_out = self.meta[self.i_in][2], self.meta[self.i_out][2]
        ck(cudart.cudaMemcpyAsync(self.dev[self.i_in], self.host[self.i_in][0], n_in, H2D, self.stream), "h2d")
        if not self.ctx.execute_async_v2(self.bindings, self.stream.value):
            raise SystemExit("execute_async_v2 failed")
        ck(cudart.cudaMemcpyAsync(self.host[self.i_out][0], self.dev[self.i_out], n_out, D2H, self.stream), "d2h")
        ck(cudart.cudaEventRecord(self.event, self.stream), "record")

    def wait(self):
        """Block until the last enqueue() finished; returns the page-locked output array, flattened."""
        ck(cudart.cudaEventSynchronize(self.event), "evsync")
        return self.host[self.i_out][1].reshape(-1)
