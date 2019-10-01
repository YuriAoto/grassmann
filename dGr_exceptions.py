"""Exceptions for dGr

"""

class dGrError(Exception):
    """Main Exception of dGr.
    
    Inherit always from this class.
    """
    def __init__(self, msg):
        self.message = msg
        
    def __str__(self):
        return str(self.message)

class dGrInputError(dGrError):
    pass

class dGrMolproInputError(dGrInputError):
    def __init__(self, msg, line=None, line_number=None):
        super().__init__(msg)
        self.line = line
        self.line_number = line_number

    def __str__(self):
        return (super().__str__() + '\n'
                + 'at line ' + str(self.line_number) + ': '
                + str(self.line))

class dGrParseError(dGrError):
    pass

class dGrValueError(dGrError, ValueError):
    pass

class dGrUnknownError(dGrError):
    def __init__(self, msg, exc_info):
        dGrError.__init__(self, msg)
        (self.exc_type,
         self.exc_value,
         self.exc_traceback) = exc_info

    def __str__(self):
        return '\n'.join(map(str, [self.message,
                                   self.exc_type,
                                   self.exc_value,
                                   self.exc_traceback]))
