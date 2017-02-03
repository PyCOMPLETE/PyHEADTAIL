# check for nvcc: if not found, do not compile the thrust module
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))

.PHONY: clean remove_so PyHEADTAIL PyHEADTAILGPU

all: PyHEADTAIL PyHEADTAILGPU

PyHEADTAIL:
	python setup.py build_ext --inplace

PyHEADTAILGPU:
ifeq ($(NVCC_TEST),nvcc)
	nvcc -Xcompiler '-fPIC' -shared -lm -o PyHEADTAIL/gpu/thrust.so PyHEADTAIL/gpu/thrust_code.cu
else
	@echo "Thrust interface not compiled because nvcc was not found"
endif

clean: remove_so
	python setup.py build_ext --inplace cleanall

remove_so:
	rm -f PyHEADTAIL/gpu/thrust.so
