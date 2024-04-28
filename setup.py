# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("main.pyx"),  # Path to your Cython file(s)
    include_dirs=[numpy.get_include()]  # Include NumPy headers
)
