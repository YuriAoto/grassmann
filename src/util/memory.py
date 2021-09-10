"""Module to control memory requirements

This module stores and handles memory requests.
This module does not really deal with memory, but stores how much
memory has been reportedly used and raise an Exception if too
much is reported.
To use it:

# first call set_total_memory, to define the maximum allowed memory.
MemoryExceededError is raised if more memory than this value is requested;

# For every amount of memory that you want to report
(usually big things), call the function allocate;

# For every amount of memory that you free, report with the
function free.

The unit given in set_total_memory is the unit that the module will use
to store the used memory. Most functions of this module will receive and return
memory in this unit. In the documentation, this unit will be denoted as UNIT

Some extra functions are also given, to help in managing the memory.

"""


_total_memory = 0.0
_used_memory = 0.0
_memory_unit = 'kB'

_maximum_allocation = 0.0
_maximum_reached = 0.0

_bytes_unit_factor = {
    'B': 0,
    'kB': 1,
    'MB': 2,
    'GB': 3,
    'TB': 4}


def convert(x, uni_in, uni_out):
    """Convert x among {,k,M,G,T}B"""
    return x * 1024**(_bytes_unit_factor[uni_in]
                      - _bytes_unit_factor[uni_out])


def mem_of_floats(n, float_size=8):
    """The amount of memory used by n floats, in UNIT"""
    return convert(n*float_size, 'B', _memory_unit)


class MemoryExceededError(Exception):
    """Exception raised when memory is exceeded"""
    
    def __init__(self, action, mem_required, mem_remaining):
        super().__init__(f'Memory limit has been exceeded:'
                         f' req={mem_required}; remaining={mem_remaining}')
        self.action = action
        self.mem_required = mem_required
        self.mem_remaining = mem_remaining


def set_total_memory(mem, unit='kB'):
    """Set the maximum allowed memory for the program
    
    Parameters:
    -----------
    mem (float)
        The maximum amount of memory
    
    unit (string: 'B', 'kB', 'MB', 'GB', 'TB')
        The unit of mem, and the unit that the module will use
    """
    if unit not in _bytes_unit_factor:
        raise ValueError('Unknown memory unit: ' + unit)
    global _total_memory
    global _memory_unit
    _total_memory = mem
    _memory_unit = unit


def allocate(mem, destination):
    """Requests extra memory.

    Parameters:
    -----------
    mem (float)
        The allocated memory, in UNIT

    destination (str)
        What the memory has been used for
    
    Raise:
    ------
    MemoryExceededError if allowed memory is exceeded
    """
    if mem > free_space():
        raise MemoryExceededError(destination, mem,
                                  free_space())
    global _used_memory
    global _maximum_allocation
    global _maximum_reached
    _used_memory += mem
    if mem > _maximum_allocation:
        _maximum_allocation = mem
    if _used_memory > _maximum_reached:
        _maximum_reached = _used_memory


def free(mem):
    """Free the requested amount of memory, in UNIT"""
    global _used_memory
    _used_memory -= mem
    if _used_memory < 0.0:
        _used_memory = 0.0


def show_status(mode='short'):
    """Return a string with information about memory usage"""
    global _maximum_allocation
    global _maximum_reached
    if mode == 'short':
        return '   Max. allocation: {1} {0}\n   Max. used: {2} {0}'.format(
            unit(), _maximum_allocation, _maximum_reached)
    if mode == 'full':
        return (f'Full memory status (in {_memory_unit}:\n'
                f'  Total:           {_total_memory}\n'
                f'  Used:            {_used_memory}\n'
                f'  Max. allocation: {_maximum_allocation}\n'
                f'  Max. used:       {_maximum_reached}')


def unit():
    """Return the UNIT"""
    return _memory_unit


def total_memory():
    """Return total allowed memory, in UNIT"""
    return _total_memory


def used_memory():
    """Return used memory, in UNIT"""
    return _used_memory


def free_space():
    """Return how much memory can still be requested, in UNIT"""
    return _total_memory - _used_memory
