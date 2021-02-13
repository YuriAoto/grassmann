"""The setup file

Usage:
------
python3 setup.py build_ext --inplace
python3 setyp.py test


"""
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("src.coupled_cluster.manifold",
              ["src/coupled_cluster/manifold.pyx"]),
    Extension("src.wave_functions.singles_doubles",
              ["src/wave_functions/singles_doubles.pyx"]),
    Extension("src.wave_functions.strings_rev_lexical_order",
              ["src/wave_functions/strings_rev_lexical_order.pyx"]),
    Extension("src.orbitals.occ_orbitals",
              ["src/orbitals/occ_orbitals.pyx"]),
    Extension("src.util.array_indices",
              ["src/util/array_indices.pyx"])
]

requires=['numpy',
          'scipy',
          'gitpython']

setup(name='grassmann',
      version='0.0',
      description=(
          'Exploring the geometry of the electronic wave functions space'),
      author='Yuri Alexandre Aoto',
      author_email='yurikungfu@gmail.com',
      tests_require=requires,
      scripts=['src/Grassmann'],
      test_suite="tests",
      ext_modules=cythonize(extensions,
                            language_level = "3",
                            include_path=['src/'],
                            annotate=True)
)
