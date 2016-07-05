"""
Location of the derived field registry
"""
import seren3

_derived_field_registry = {}  # Automatically filled by annotations

_pynbody_to_pymses_registry = {"Msol" : "Msun"}  # translate pynbody units to pymses

# Warning, do not include epoch in here
_tracked_field_unit_registry = {"rho" : {"info_key" : "unit_density"}, \
                                "vel" : {"info_key" : "unit_velocity"}, \
                                "P" : {"info_key" : "unit_pressure"}, \
                                "dx" : {"info_key" : "unit_length"}, \
                                # "Np" : {"info_key" : "unit_photon_number", "unit" : "m**-3"}, \
                                "Np" : {"info_key" : "unit_photon_flux_density"}, \
                                "Fp" : {"info_key" : "unit_photon_flux_density"}, \
                                "pos" : {"info_key" : "unit_length"}, \
                                "mass" : {"info_key" : "unit_mass", "default_unit" : "Msol"}}


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

def in_tracked_field_registry(field):
    return field in _tracked_field_unit_registry

def info_for_tracked_field(field):
    return _tracked_field_unit_registry[field]

def derived_quantity(requires, unit):
    def wrap(fn):
        _derived_field_registry[fn.__name__] = fn
        _derived_field_registry["%s_required" % fn.__name__] = requires
        _derived_field_registry["%s_unit" % fn.__name__] = unit
        return fn
    return wrap

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