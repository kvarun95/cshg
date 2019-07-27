from distutils.core import setup, Extension
import numpy

# g++ -o cfdct3d cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw
cpp_compile_args = ["-fPIC", "-L/home/varun/fftw-2.1.5/fftw/.libs", "-lfftw"]
include_dirs=[numpy.get_include(), '.']

cfdct3d_module = Extension('_cfdct3d',
                           sources=['cfdct3d.cpp',
                                    # 'fdct3d_forward.cpp',
                                    # 'fdct3d_inverse.cpp',
                                    # 'fdct3d_param.cpp',
                                    'cfdct3d.i'],
                           swig_opts=["-c++"],
                           extra_compile_args=cpp_compile_args,
                           extra_objects=["libfdct3d.a"],
                           include_dirs=include_dirs,
                           library_dirs=['/home/varun/fftw-2.1.5/fftw/.libs', '.'],
                           libraries=['fftw'],
                                    )


setup(name='cfdct3d',
      version='0.1',
      ext_modules=[cfdct3d_module],
      py_modules=['cfdct3d'],
      )


