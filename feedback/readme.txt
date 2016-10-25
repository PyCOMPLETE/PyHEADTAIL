# Readme

This is a simplified version of the feedback module, which main repository can be found from
[gitlab.cern.ch/jakomppu/PyHEADTAIL_feedback](https://gitlab.cern.ch/jakomppu/PyHEADTAIL_feedback).
The code here can be found as a branch in the main repository of the module.

## v0.1.1 -> v0.1.2  (25.10.2016)
    * The previous solution to the np.dot performance issue in LSF did not work with all python versions. Thus in this
      version matrix product has been written in Cython which seems to be a stable solution to the problem.

## v0.1.1  (28.09.2016)
    * The first version included into PyHEADTAIL








