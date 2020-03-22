import cv2
import pyopencl as cl
from pyopencl.tools import get_test_platforms_and_devices

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
cv2.OPENCV_OPENCL_DEVICE = 'AMD:GPU'

print(get_test_platforms_and_devices())

print("OpenCL Available: ", cv2.ocl.haveOpenCL())

print(cv2.ocl_Device().getDefault())

print(cv2.ocl_Device().isAMD())
print(cv2.ocl_Device().isIntel())
