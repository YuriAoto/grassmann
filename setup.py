"""The setup file

Usage:
------
python3 setup.py build_ext --inplace
python3 setup.py test


"""
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("src.integrals.integrals_cy",
              ["src/integrals/integrals_cy.pyx"]),
    Extension("src.coupled_cluster.manifold",
              ["src/coupled_cluster/manifold.pyx"]),
    Extension("src.wave_functions.general",
              ["src/wave_functions/general.pyx"]),
    Extension("src.wave_functions.fci",
              ["src/wave_functions/fci.pyx"]),
    Extension("src.wave_functions.interm_norm",
              ["src/wave_functions/interm_norm.pyx"]),
    Extension("src.wave_functions.cisd",
              ["src/wave_functions/cisd.pyx"]),
    Extension("src.wave_functions.singles_doubles",
              ["src/wave_functions/singles_doubles.pyx"]),
    Extension("src.wave_functions.strings_rev_lexical_order",
              ["src/wave_functions/strings_rev_lexical_order.pyx"]),
    Extension("src.orbitals.occ_orbitals",
              ["src/orbitals/occ_orbitals.pyx"]),
    Extension("src.orbitals.orbital_space",
              ["src/orbitals/orbital_space.pyx"]),
    Extension("src.util.array_indices",
              ["src/util/array_indices.pyx"]),
    Extension("src.integrals.integrals_cy",
              ["src/integrals/integrals_cy.pyx"])
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
