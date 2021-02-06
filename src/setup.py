"""Setup to compile all cython modules

Usage:
------
python3 setup.py build_ext --inplace


"""

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("coupled_cluster.manifold",
              ["coupled_cluster/manifold.pyx"]),
    Extension("wave_functions.strings_rev_lexical_order",
              ["wave_functions/strings_rev_lexical_order.pyx"]),
    Extension("util.array_indices",
              ["util/array_indices.pyx"])
]

setup(ext_modules=cythonize(extensions,
                            language_level = "3",
                            annotate=True))
