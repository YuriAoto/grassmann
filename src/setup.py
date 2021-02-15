"""Setup to compile all cython modules

Usage:
------
python3 setup.py build_ext --inplace


"""

from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
#    Extension("coupled_cluster.manifold",
#              ["coupled_cluster/manifold.pyx"]),
    Extension("integrals.integrals_cy",
              ["integrals/integrals_cy.pyx"]),
    Extension("wave_functions.strings_rev_lexical_order",
              ["wave_functions/strings_rev_lexical_order.pyx"]),
    Extension("tests.implementation.residual_cy",
              ["../tests/implementation/residual_cy.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]

setup(ext_modules=cythonize(extensions,
                            language_level = "3",
                            annotate=True))
