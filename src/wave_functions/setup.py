from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("strings_rev_lexical_order", ["strings_rev_lexical_order.pyx"])
]

setup(ext_modules=cythonize(extensions,
                            language_level = "3",
                            annotate=True))
