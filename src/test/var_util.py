import logging

import memory


init_random_state = 1234

log_format = ('%(levelname)s: %(funcName)s - %(filename)s:'
              + '\n%(message)s\n')
logging.basicConfig(filename='testing.log',
                    format=log_format,
                    filemode='w',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


memory.set_total_memory(100.0)
