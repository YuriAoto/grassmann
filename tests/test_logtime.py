import os
import sys
import logging
from time import sleep

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from dGr_util import logtime


logfile = 'testing_logtime.log'
log_format = ('%(levelname)s: %(funcName)s - %(filename)s:'
              + '\n%(message)s\n')
logging.basicConfig(filename=logfile,
                    format=log_format,
                    filemode='w',
                    level=logging.INFO)


with logtime('Testing logtime') as T:
    print('Inside test logtime')
print('Elapsed time (outside with): ', T.elapsed_time)


with logtime('Testing logtime, with sleep 3 and stdout', f_out=sys.stdout) as T:
    print('Inside test logtime')
    sleep(3.0)
print('Elapsed time (outside with): ', T.elapsed_time)


with logtime('Testing logtime, with sleep 3 and stdout, without as T', f_out=sys.stdout):
    print('Inside test logtime')
    sleep(3.0)


with logtime('Testing logtime, with sleep 3 and stdout, without as T, with format',
             f_out=sys.stdout, out_fmt='Elapsed time: {}\n'):
    print('Inside test logtime')
    sleep(3.0)
