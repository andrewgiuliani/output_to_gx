# output_to_gx
This is a utility to compute geometric quantities of devices in QUASR for gyrokinetic simulations.  To install:

    git clone --recursive
    cd pycheb/
    pip install -e .
    cd ..
    
    pip install ground bentley_ottmann

SIMSOPT also is assumed to be installed.  The unit tests reveal proper usage of the `output_to_gx` function.

Unit tests can be run by calling:
    python -m unittest test_output_to_gx.py
