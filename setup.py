"""The setup file

Usage:
------
python3 setup.py build_ext --inplace
python3 setup.py test


"""
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

#
# Cython files (without the .pyx extension)
#
all_cython_files = ["src/integrals/integrals_cy",
                    "src/coupled_cluster/exc_on_string",
                    "src/coupled_cluster/manifold",
                    "src/coupled_cluster/manifold_term1",
                    "src/coupled_cluster/manifold_term2",
                    "src/coupled_cluster/manifold_hess",
                    "src/wave_functions/general",
                    "src/wave_functions/fci",
                    "src/wave_functions/interm_norm",
                    "src/wave_functions/cisd",
                    "src/wave_functions/strings_rev_lexical_order",
                    "src/orbitals/occ_orbitals",
                    "src/orbitals/orbital_space",
                    "src/util/array_indices",
                    "src/coupled_cluster/cluster_decomposition"]


extensions = []
for cy_file in all_cython_files:
    extensions.append(Extension(cy_file.replace('/', '.'),
                                [cy_file + '.pyx']))

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
