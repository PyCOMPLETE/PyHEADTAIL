all: PyHEADTAIL

PyHEADTAIL:
	python setup.py build_ext --inplace

clean:
	python setup.py build_ext --inplace cleanall
