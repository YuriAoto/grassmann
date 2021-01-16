from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("manifold", ["manifold.pyx"])
]

setup(ext_modules=cythonize(extensions,
                            language_level = "3",
                            annotate=True))
