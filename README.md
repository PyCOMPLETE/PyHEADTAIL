# PyHEADTAIL

CERN `PyHEADTAIL` is a macro-particle code for simulating beam dynamics in particle accelerators with collective effects.

`PyHEADTAIL` is a `Python3.6+` code written with some C extensions, compiled with `Cython`.

## Installing

`PyHEADTAIL` is available directly from `PyPI` and can be installed in your desired environment with:
```bash
python -m pip install PyHEADTAIL
```

Once installed, users can directly import and start using `PyHEADTAIL`:
```python
import PyHEADTAIL

# Whatever you wish to use PyHEADTAIL for
```

Example scripts with basic usage can be found in the `examples` folder of this repository.

<details>
  <summary>Note: Use of OpenMP</summary>

By default `PyHEADTAIL` extensions are compiled with `OpenMP` support.
This can be especially troublesome for `macOS` users, as `OpenMP` support on the platform is possible but tricky.

To simplify installation, one can disable the use `OpenMP` at compilation by simply setting the `PYHT_USE_OPEN_MP` environment variable to `0` before running the above installation command.
This is as simple as:
```bash
export PYHT_USE_OPENMP=0
```
</details>
<br/>

<details>
  <summary>Note: GPU Support</summary>

For GPU usage support, no wheel or option is available directly from `PyPI`.
Instead, the developer version is required (a `Makefile` is included in the source code).
See the `Contributing` section below for instructions.
</details>

## Contributing

For developers of `PyHEADTAIL`, we recommend an [editable install][pip_editable] from VCS.

**Note:** 

We recommend using a `conda` environment for isolation.
Having git available, installation is straight forward:

1. Clone the repository in a local folder:
```bash
git clone https://github.com/PyCOMPLETE/PyHEADTAIL
cd PyHEADTAIL
```

2. Create an environment and get the necessary dependencies. With conda, this goes as:
```bash
conda create -n pyheadtail -c conda-forge numpy scipy h5py cython
conda activate pyheadtail
```

3. Go to the folder and run the installation script:

|   Standalone Install - Attempts GPU Support   | Editable Install                      |
| :-------------------------------------------: | :-----------------------------------: |
| `make` (see `Makefile` for available targets) | `python -m pip install --editable .`  |

1. Start using `PyHEADTAIL`:
```python
import PyHEADTAIL

# Now test your changes!
```

## License

This project is licensed under the `BSD-3 License` - see the [LICENSE](LICENSE.txt) file for details.

[pip_editable]: https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs