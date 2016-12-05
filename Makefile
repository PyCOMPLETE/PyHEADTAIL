# check for nvcc: if not found, do not compile the thrust module
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))


all: PyHEADTAIL PyHEADTAILGPU #errfff

PyHEADTAIL:
	python setup.py build_ext --inplace

PyHEADTAILGPU:
ifeq ($(NVCC_TEST),nvcc)
	nvcc -Xcompiler '-fPIC' -shared -lm -o gpu/thrust.so gpu/thrust_code.cu
else
	@echo "Thrust interface not compiled because nvcc was not found"
endif

#errfff:
#	f2py -c general/errfff.f -m errfff
#	mv errfff.so general/

clean: remove_so
	python setup.py build_ext --inplace cleanall

remove_so:
	rm -f gpu/thrust.so
