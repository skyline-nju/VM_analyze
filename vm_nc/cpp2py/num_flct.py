# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

"""Cal number fluctuations."""


from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_num_flct')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_num_flct')
    _num_flct = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_num_flct', [dirname(__file__)])
        except ImportError:
            import _num_flct
            return _num_flct
        try:
            _mod = imp.load_module('_num_flct', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _num_flct = swig_import_helper()
    del swig_import_helper
else:
    import _num_flct
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0


def cal_num_flct_3(num, box_len, box_num, num_mean, num_var):
    """cal_num_flct_3(unsigned short * num, int * box_len, int * box_num, double * num_mean, double * num_var)"""
    return _num_flct.cal_num_flct_3(num, box_len, box_num, num_mean, num_var)

def renormalize_2d_uint16(arg1, out):
    """renormalize_2d_uint16(unsigned short * arg1, int * out)"""
    return _num_flct.renormalize_2d_uint16(arg1, out)

def renormalize_2d_doub(arg1, out):
    """renormalize_2d_doub(double * arg1, double * out)"""
    return _num_flct.renormalize_2d_doub(arg1, out)

def cal_num_flct(num, box_len, box_num, num_mean, num_var, box_len_dim):
    """cal_num_flct(unsigned short * num, int * box_len, int * box_num, double * num_mean, double * num_var, int box_len_dim)"""
    return _num_flct.cal_num_flct(num, box_len, box_num, num_mean, num_var, box_len_dim)
# This file is compatible with both classic and new-style classes.


