import pyximport;
pyximport.install(setup_args = {"script_args" : ["--force"]},
                  language_level=3)

# now drag CyTester into the global namespace, 
# so tests can be discovered by unittest
from tests.unit.test_cc_manifold import *
from tests.unit.test_occ_orbitals import *
