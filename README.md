# Compressive Second Harmonic Generation Imaging

3D Coherent second harmonic generation imaging based on wirtinger flow based phase retrieval from phase coded aperture patterns, followed by solving a regularized linear inverse problem.

## Installation and use:

- Install fftw2.
- Then install the included curvelab package as follows:

    Enter `/Curvelab/`.
    Edit `makefile.opt` for specify the directory in which fftw is installed.
    Create libraries using 
    ```
    make lib
    ```
    Test using 
    ```
    make test
    ```
- Enter `/Curvelab/fdct3d/src/`.

    Check if 3d curvelet transform is working fine in C++:
    ```
    make test
    ./test options
    ```
    In `setup.py`, set `FFTW` to the path to the fftw installation directory. (same as the one used previously).
    Install the python package which wraps fdct3d using cython:
    ```
    bash installer.sh cython
    ```

    For testing, run functions from the file `test.py`.
