#cython: language_level=3
from distutils.core import setup
from Cython.Build import cythonize

setup(name="cutsky_fast", ext_modules=cythonize("cutsky_fast.pyx"),)
