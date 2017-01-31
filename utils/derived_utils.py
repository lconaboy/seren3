"""
Location of the derived field registry
"""
import seren3
from seren3.array import SimArray
from seren3.core.serensource import DerivedDataset

_derived_field_registry = {}  # Automatically filled by annotations

_pynbody_to_pymses_registry = {"Msol" : "Msun"}  # translate pynbody units to pymses

# Warning, do not include epoch in here
_tracked_field_unit_registry = {"rho" : {"info_key" : "unit_density", "latex" : r"$\rho$"}, \
                                "vel" : {"info_key" : "unit_velocity", "latex" : r"$\vec{v}$"}, \
                                "P" : {"info_key" : "unit_pressure"}, \
                                "dx" : {"info_key" : "unit_length", "latex" : r"$\Delta$x"}, \
                                "size" : {"info_key" : "unit_length", "latex" : r"$\Delta$x"}, \
                                # "Np" : {"info_key" : "unit_photon_number", "unit" : "m**-3"}, \
                                "Np" : {"info_key" : "unit_photon_flux_density"}, \
                                "Fp" : {"info_key" : "unit_photon_flux_density"}, \
                                # "xHII" : {"info_key" : None}, \
                                # "xHeII" : {"info_ley" : None}, \
                                # "xHeIII" : {"info_key" : None}, \
                                "pos" : {"info_key" : "unit_length"}, \
                                "mass" : {"info_key" : "unit_mass", "default_unit" : "Msol", "latex" : "M"}}

def pymses_units(unit_string):
    '''
    Returns pymses compatible units from a string
    '''
    import numpy as np
    from pymses.utils import constants as C
    unit = 1.
    compontents = str(unit_string).split(' ')
    for c in compontents:
        if '**' in c:
            dims = c.split('**')
            pymses_unit = np.power(C.Unit(dims[0]), float(dims[1]))
            unit *= pymses_unit
        else:
            if c in _pynbody_to_pymses_registry:
                c = _pynbody_to_pymses_registry[c]
            unit *= C.Unit(c)
    return unit


def get_field_unit(context, field):
    '''
    Returns default unit for this field
    '''
    if hasattr(context, "family") and (is_derived(context.family, field)):
        return get_derived_field_unit(context.family, field)
    elif in_tracked_field_registry(field):
        field_info = info_for_tracked_field(field)
        if "default_unit" in field_info:
            return pymses_units(field_info["default_unit"])
        elif field_info["info_key"] is not None:
            return context.info[field_info["info_key"]]
    else:
        return context.C.none


def in_tracked_field_registry(field):
    return field in _tracked_field_unit_registry


def info_for_tracked_field(field):
    if field[-1].isdigit():
        return _tracked_field_unit_registry[field[:-1]]
    return _tracked_field_unit_registry[field]

def derived_quantity(requires, unit):
    def _check_dset(fn, context, dset, **kwargs):
        '''
        Ensures a DerivedDataset is also handed off to the worker function
        '''
        if not isinstance(dset, DerivedDataset):
            dset = DerivedDataset(context, dset)
        return fn(context, dset, **kwargs)
    def wrap(fn):
        _derived_field_registry[fn.__name__] = lambda context, dset, **kwargs: _check_dset(fn, context, dset, **kwargs) #fn
        _derived_field_registry["%s_required" % fn.__name__] = requires
        _derived_field_registry["%s_unit" % fn.__name__] = unit
        return fn
    return wrap

# def derived_quantity(requires, unit):
#     # Ensures fields are SimArrays with correct unit information
#     def _check_dset(fn, context, dset, **kwargs):
#         parsed_dset = {}
#         keys = []
#         if isinstance(dset, dict):
#             keys = dset.keys()
#         elif hasattr(dset, "fields"):
#             keys = dset.fields
#             if hasattr(dset, "get_sizes"):
#                 parsed_dset["dx"] = SimArray(dset.get_sizes(), context.info["unit_length"])
#             if hasattr(dset, "points"):
#                 parsed_dset["pos"] = SimArray(dset.points, context.info["unit_length"])
#         else:
#             raise Exception("Can't get keys for dset: %s" % dset)

#         for field in keys:
#             if not isinstance(dset[field], SimArray):
#                 field_info = None
#                 f = None
#                 if field[-1].isdigit():
#                     # field_info = info_for_tracked_field(field[:-1])
#                     f = field[:-1]
#                 else:
#                     f = field
#                     # field_info = info_for_tracked_field(field)
#                 if in_tracked_field_registry(f):
#                     field_info = info_for_tracked_field(f)
#                     unit_key = field_info["info_key"]

#                     unit = context.info[unit_key]
#                     parsed_dset[field] = SimArray(dset[field], unit)
#                     if "default_unit" in field_info:
#                         parsed_dset[field] = parsed_dset[field].in_units(field_info["default_unit"])
#                     if "latex" in field_info:
#                         parsed_dset[field].set_field_latex(field_info["latex"])
#                 else:
#                     # parsed_dset[field] = SimArray(dset[field], 1)
#                     # parsed_dset[field].set_latex(field)
#                     parsed_dset[field] = dset[field]
#             else:
#                 parsed_dset[field] = dset[field]

#         print parsed_dset.keys()
#         return fn(context, parsed_dset, **kwargs)
#     def wrap(fn):
#         _derived_field_registry[fn.__name__] = lambda context, dset, **kwargs: _check_dset(fn, context, dset, **kwargs) #fn
#         _derived_field_registry["%s_required" % fn.__name__] = requires
#         _derived_field_registry["%s_unit" % fn.__name__] = unit
#         return fn
#     return wrap


def add_derived_quantity(fn, requires):
    _derived_field_registry[fn.__name__] = fn
    _derived_field_registry["%s_required" % fn.__name__] = requires


def required_for_field(family, field):
    return _derived_field_registry["%s_%s_required" % (family, field)]


def is_derived(family, field):
    return "%s_%s" % (family, field) in _derived_field_registry


def get_derived_field(family, field):
    return _derived_field_registry["%s_%s" % (family, field)]


def get_derived_field_unit(family, field):
    return _derived_field_registry["%s_%s_unit" % (family, field)]


def LambdaOperator(family, field, power=1., vol_weighted=False):
    '''
    Return a lambda function for this field
    '''
    # If this is a derived field then grab the approp. function
    if is_derived(family.family, field):
        fn = get_derived_field(family.family, field)
        if vol_weighted:
            op = lambda dset: fn(family.base, dset)**power * dset.get_sizes()**3
        else:
            op = lambda dset: fn(family.base, dset)**power
    else:
        if vol_weighted:
            op = lambda dset: dset[field]**power * dset.get_sizes()**3
        else:
            op = lambda dset: dset[field]**power
    return op
