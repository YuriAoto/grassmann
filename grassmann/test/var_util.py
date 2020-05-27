import logging

init_random_state = 1234

log_format = ('%(levelname)s: %(funcName)s - %(filename)s:'
              + '\n%(message)s\n')
logging.basicConfig(filename='testing.log',
                    format=log_format,
                    filemode='a',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

main_files_dir = ('/home/yuriaoto/Documents/Codes/min_dist_Gr/'
                  + 'grassmann/test/inputs_outputs/')

files_dir = {'H2_sto3g_D2h': 'H2__R_5__sto3g__D2h/',
             'He2_631g_D2h': 'He2__R_1.5__631g__D2h/',
             'He2_631g_C2v': 'He2__R_1.5__631g__C2v/',
             'Li2_631g_C2v': 'He2__R_5__631g__C2v/'
             }


def xml_orb_file(label):
    return main_files_dir + files_dir[label] + 'orbitals.xml'


def RHF_file(label):
    return main_files_dir + files_dir[label] + 'RHF.out'


def CISD_file(label):
    return main_files_dir + files_dir[label] + 'CISD.out'


def FCI_file(label):
    return main_files_dir + files_dir[label] + 'FCI.out'
