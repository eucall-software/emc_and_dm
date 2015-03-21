from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as N

exMod = [Extension("rotateIntens", ["rotateIntens.pyx"], extra_compile_args=['-O2'], extra_link_args=[])]
setup(name = 'myFunctions', cmdclass = {'build_ext': build_ext}, include_dirs = [N.get_include()], ext_modules = exMod)

#To install in place:
#python setup.py build_ext --inplace
