.PHONY: clean PyHEADTAIL PyHEADTAILGPU errfff

all: PyHEADTAIL errfff

PyHEADTAIL:
	python setup.py build_ext --inplace

#errfff:
#	f2py -c PyHEADTAIL/general/errfff.f90 -m errfff
#	mv errfff.so PyHEADTAIL/general/

clean:
	python setup.py build_ext --inplace cleanall
