from ctypes import *
import numpy as np
import numpy.ctypeslib as nc
import os
import platform

if os.path.exists(os.getcwd() + '\\lib'):
    os.add_dll_directory(os.getcwd() + '\\lib')

os_name = platform.system()
if os_name == 'Linux':
    path = '../lib/libSoyNet.so'
elif os_name == 'Windows':
    path = '../lib/SoyNet.dll'


if os.path.exists(path):
    lib = cdll.LoadLibrary(path)
else:
    print("Can't find SoyNet.dll")
    exit(-1)


lib.initSoyNet.argtypes = [c_char_p, c_char_p]
lib.initSoyNet.restype = c_void_p
def initSoyNet_f32(cfg, extend_param="") :
    if extend_param is None: extend_param = ""
    return lib.initSoyNet(cfg.encode("utf8"), extend_param.encode("utf8"))


F32 = nc.ndpointer(dtype=np.float32, ndim=1, flags='aligned, c_contiguous')
lib.feedData.argtypes=[c_void_p, F32]
lib.feedData.restype=None
def feedData_f32(handle, data) :
    lib.feedData(handle, data.flatten())


lib.inference.argtypes=[c_void_p]
lib.inference.restype=None
inference = lib.inference


# class OUT(Structure): _fields_=[("f", c_float)]
#
# getOutput=lib.getOutput
# getOutput.argtypes=[c_void_p, POINTER(OUT)]
# getOutput.restype=None

F32 = nc.ndpointer(dtype=np.float32, ndim=3, flags='aligned, c_contiguous')
lib.getOutput.argtypes=[c_void_p, F32]
lib.getOutput.restype=None
def getOutput_f32(handle, output) :
    lib.getOutput(handle, output)


lib.freeSoyNet.argtypes=[c_void_p]
lib.freeSoyNet.restype=None
def freeSoyNet_f32(handle) :
    lib.freeSoyNet(handle)
