# check for nvcc: if not found, do not compile the thrust module
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))


all: PyHEADTAIL PyHEADTAILGPU

PyHEADTAIL:
	python setup.py build_ext --inplace

PyHEADTAILGPU: gpu/thrust_code.cu gpu/thrust.so
ifeq ($(NVCC_TEST),nvcc)
	nvcc -Xcompiler '-fPIC' -shared -o gpu/thrust.so gpu/thrust_code.cu
else
	@echo "Thrust interface not compiled because nvcc was not found"
endif
clean:
	python setup.py build_ext --inplace cleanall
