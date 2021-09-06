import pyximport;
pyximport.install(setup_args = {"script_args" : ["--force"]},
                  language_level=3)

# now drag CyTester into the global namespace, 
# so tests can be discovered by unittest
from tests.unit.test_cc_exc_on_str import *
from tests.unit.test_cc_cluster_dec_cy import *
from tests.unit.test_cc_manifold import *
from tests.unit.test_orbital_space import *
