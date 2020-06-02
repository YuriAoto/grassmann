import logging

init_random_state = 1234

log_format = ('%(levelname)s: %(funcName)s - %(filename)s:'
              + '\n%(message)s\n')
logging.basicConfig(filename='testing.log',
                    format=log_format,
                    filemode='a',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

main_files_dir = ('/home/yuriaoto/Documents/Codes/min_dist_Gr/'
                  + 'grassmann/test/inputs_outputs/')


def xml_orb_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/orbitals.xml'


def RHF_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/RHF.out'


def CISD_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/CISD.out'


def CCSD_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/CCSD.out'


def FC_CISD_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/FC_CISD.out'


def FCI_file(inp_out_dir):
    return main_files_dir + inp_out_dir + '/FCI.out'
