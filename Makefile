# check for nvcc: if not found, do not compile the thrust module
NVCC_RESULT := $(shell which nvcc)
NVCC_TEST := $(notdir $(NVCC_RESULT))

.PHONY: clean PyHEADTAIL PyHEADTAILGPU errfff

all: PyHEADTAIL PyHEADTAILGPU errfff

PyHEADTAIL:
	python setup.py build_ext --inplace

PyHEADTAILGPU:
ifeq ($(NVCC_TEST),nvcc)
	nvcc -Xcompiler '-fPIC' -shared -lm -o PyHEADTAIL/gpu/thrust.so PyHEADTAIL/gpu/thrust_code.cu
else
	@echo "GPU: Thrust interface not compiled because nvcc compiler not found."
endif

errfff:
	f2py -c PyHEADTAIL/general/errfff.f90 -m errfff
	mv errfff*.so PyHEADTAIL/general/

clean:
	python setup.py build_ext --inplace cleanall
	rm -f PyHEADTAIL/gpu/thrust.so
