from distutils.sysconfig import get_config_vars as default_get_config_vars
import distutils.sysconfig as dsc
from distutils.core import setup, Extension
import numpy
from Cython.Build import cythonize

def remove_pthread(x):
    if type(x) is str:
        # x.replace(" -pthread ") would be probably enough...
        # but we want to make sure we make it right for every input
        if x=="-pthread":
            return ""
        if x.startswith("-pthread "):
            return remove_pthread(x[len("-pthread "):])
        if x.endswith(" -pthread"):
            return remove_pthread(x[:-len(" -pthread")])
        return x.replace(" -pthread ", " ")
    return x

def my_get_config_vars(*args):
  result = default_get_config_vars(*args)
  # sometimes result is a list and sometimes a dict:
  if type(result) is list:
     return [remove_pthread(x) for x in result]
  elif type(result) is dict:
     return {k : remove_pthread(x) for k,x in result.items()}
  else:
     raise Exception("cannot handle type"+type(result))

# 2.step: replace    
dsc.get_config_vars = my_get_config_vars

SWIG = False

if SWIG:
      # g++ -o cfdct3d cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw
      cpp_compile_args = ["-fPIC", "-L/home/varun/fftw-2.1.5/fftw/.libs", "-lfftw"]
      include_dirs=[numpy.get_include(), '.', '/home/varun/fftw-2.1.5/fftw/.libs']

      cfdct3d_module = Extension('_cfdct3d',
                              sources=['cfdct3d.cpp',
                                          # 'fdct3d_forward.cpp',
                                          # 'fdct3d_inverse.cpp',
                                          # 'fdct3d_param.cpp',
                                          'cfdct3d.i'],
                              swig_opts=["-c++"],
                              extra_compile_args=cpp_compile_args,
                              extra_objects=["libfdct3d.a", "/home/varun/fftw-2.1.5/fftw/.libs/libfftw.a"],
                              include_dirs=include_dirs,
                              library_dirs=['/home/varun/fftw-2.1.5/fftw/.libs', '.'],
                              libraries=['fftw', 'fdct3d'],
                                          )


      setup(name='cfdct3d',
            version='0.1',
            ext_modules=[cfdct3d_module],
            py_modules=['cfdct3d'],
            )
else:
      # g++ -o cfdct3d cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw
      cpp_compile_args = ["-fPIC", "-L/home/varun/fftw-2.1.5/fftw/.libs", "-lfftw"]
      include_dirs=[numpy.get_include(), '.', '/home/varun/fftw-2.1.5/fftw/.libs']

      cfdct3d_module = Extension(name='pycfdct3d',
                              sources=['pycfdct3d.pyx', 'cfdct3d.cpp'],
                              extra_compile_args=cpp_compile_args,
                              extra_objects=["libfdct3d.a", "/home/varun/fftw-2.1.5/fftw/.libs/libfftw.a"],
                              include_dirs=include_dirs,
                              library_dirs=['/home/varun/fftw-2.1.5/fftw/.libs', '.'],
                              libraries=['fftw', 'fdct3d'],
                                          )


      setup(name='pycfdct3d',
            version='0.1',
            ext_modules=cythonize([cfdct3d_module], language='c++'),
            # py_modules=['cfdct3d'],
            )

