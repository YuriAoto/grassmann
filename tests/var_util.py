"""Useful variables for tests. See package documentation"""
import os
import logging
import unittest

from util import memory
from input_output.log import loglevel_from_str

init_random_state = 1234

log_format = ('%(levelname)s: %(funcName)s - %(filename)s:'
              + '\n%(message)s\n')
logging.basicConfig(filename='testing.log',
                    format=log_format,
                    filemode='w',
                    level=loglevel_from_str(os.getenv('GR_LOGLEVEL')))
logger = logging.getLogger(__name__)


_GR_MEMORY_env = os.getenv('GR_MEMORY')
memory.set_total_memory(400.0
                        if _GR_MEMORY_env is None else
                        float(_GR_MEMORY_env))

all_test_categories = [
    'ALL',
    'ESSENTIAL',
    'VERY SHORT',
    'SHORT',
    'LONG',
    'VERY LONG',
    'COMPLETE',
    'PROFILE',
    'NONE'
    ]


_GR_TESTS_GROUP_env = os.getenv('GR_TESTS_CATEG')
if _GR_TESTS_GROUP_env is None:
    user_categories = ('ALL',)
else:
    user_categories = tuple(_GR_TESTS_GROUP_env.split(','))
for cat in user_categories:
    if cat not in all_test_categories:
        raise Exception(cat + ' is not a valid test category!')
run_all_categories = 'ALL' in user_categories
if run_all_categories:
    print('Run all test categories!')
else:
    print(f'Run only the following test categories: {", ".join(user_categories)}')


def _is_in_user_categories(test_categories):
    for cat in test_categories:
        if cat not in all_test_categories:
            raise Exception(cat + ' is not a valid test category!')
    if run_all_categories:
        return True
    for cat in test_categories:
        if cat in user_categories:
            return True
    return False


def category(*cats):
    if _is_in_user_categories(cats):
        return lambda func: func
    return unittest.skip(f'Test category: {", ".join(cats)}')
