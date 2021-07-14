"""Logging

"""
import logging
import time
from datetime import timedelta

logger = logging.getLogger(__name__)

_loglevels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG,
              'notset': logging.NOTSET}


def loglevel_from_str(x):
    """Transform a string to a loglevel"""
    try:
        return int(x)
    except ValueError:
        try:
            return _loglevels[x.lower()]
        except KeyError:
            raise ParseError(f'This is not a valid log level: {x}')


class logtime():
    """A context manager for logging execution time.
    
    Examples:
    ----------
    with logtime('Executing X'):
        # Add time to log (with level INFO)
        
    with logtime('Executing X', log_level=logging.DEBUG):
        # Add time to log (with level DEBUG)
    
    with logtime('Executing X', out_stream=sys.stdout):
        # Add time to sys.stdout as well
    
    with logtime('Executing X',
                 out_stream=sys.stdout,
                 out_fmt="It took {} to run X"):
        # Use out_fmt to write elapsed time to sys.stdout
    
    with logtime('Executing X') as T_X:
        # Save info in object T_X
    print(T_X.elapsed_time)
    
    with logtime('Executing X') as T_X:
        # Save info in object T_X
    with logtime('Executing X') as T_Y:
        # Save info in object T_Y
    print('Time for X and Y: ',
          timedelta(seconds=(T_Y.end_time - T_X.ini_time)))
    """
    def __init__(self,
                 action_type,
                 log_level=logging.INFO,
                 out_stream=None,
                 out_fmt=None):
        self.action_type = action_type
        self.log_level = log_level
        self.out_stream = out_stream
        self.end_time = None
        self.elapsed_time = None
        if out_fmt is None:
            self.out_fmt = 'Elapsed time for ' + self.action_type + ': {}\n'
        else:
            self.out_fmt = out_fmt
    
    def __enter__(self):
        self.ini_time = time.time()
        logger.log(self.log_level,
                   '%s ...',
                   self.action_type)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.time()
        self.elapsed_time = str(timedelta(seconds=(self.end_time
                                                   - self.ini_time)))
        logger.info('Total time for %s: %s',
                    self.action_type,
                    self.elapsed_time)
        if self.out_stream is not None:
            self.out_stream.write(self.out_fmt.format(self.elapsed_time))
    
    def relative_to(self, other):
        """Return the time from other.ini_time to self.end_time"""
        return str(timedelta(seconds=(self.end_time - other.ini_time)))


class LogFilter():
    """Define a filter for logging, to be attached to all handlers."""
    def __init__(self, logfilter_re):
        """Initialises the class
        
        Parameters:
        -----------
        logfilter_re (str, with a regular expression)
            Only functions that satisfy this RE will be logged
        """
        self.logfilter_re = logfilter_re
    
    def filter(self, rec):
        """Return a boolean, indicating whether rec will be logged or not."""
        if rec.funcName == '__enter__':
            rec.funcName = 'Entering time management'
        elif rec.funcName == '__exit__':
            rec.funcName = 'Finishing time management'
        if self.logfilter_re is not None:
            return self.logfilter_re.search(rec.funcName) is not None
        # Uncomment this to check for possible records to filter
        # print()
        # print(rec)
        # print(rec.__dict__)
        # print()
        return True
