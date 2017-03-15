# check for nvcc: if not found, do not compile the thrust module
NVCC_RESULT := $(shell which nvcc)
NVCC_TEST := $(notdir $(NVCC_RESULT))

.PHONY: clean PyHEADTAIL PyHEADTAILGPU

all: PyHEADTAIL PyHEADTAILGPU

PyHEADTAIL:
	python setup.py build_ext --inplace

PyHEADTAILGPU:
ifeq ($(NVCC_TEST),nvcc)
	nvcc -Xcompiler '-fPIC' -shared -lm -o PyHEADTAIL/gpu/thrust.so PyHEADTAIL/gpu/thrust_code.cu
else
	@echo "GPU: Thrust interface not compiled because nvcc compiler not found."
endif

clean:
	python setup.py build_ext --inplace cleanall
	rm -f PyHEADTAIL/gpu/thrust.so
