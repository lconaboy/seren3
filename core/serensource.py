import seren3
from pynbody.array import SimArray
from pymses.sources.ramses.sources import RamsesAmrSource, RamsesParticleSource
from pymses.core import sources

class SerenSource(sources.DataSource):
    """
    Class to extend pymses source and implement derived fields
    """
    def __init__(self, family, source, required_fields, requested_fields, cpu_list=None):
        super(SerenSource, self).__init__()
        self._source = source
        self._family = family
        self._dset = None
        self._cpu_list = cpu_list
        self.required_fields = required_fields
        self.requested_fields = requested_fields

    def __iter__(self):
        cpu_list = None
        if self._cpu_list is not None:
            cpu_list = self._cpu_list
        else:
            cpu_list = range(1, self._family.info['ncpu'] + 1)
        for idomain in cpu_list:
            yield self.get_domain_dset(idomain)

    @property
    def points(self):
        if self._dset is None:
            return self._source.flatten().points
        return self._dset.points

    def get_sizes(self):
        if self._dset is None:
            return self._source.flatten().get_sizes()
        return self._dset.get_sizes()

    @property
    def source(self):
        '''
        The base RamsesAmrSource
        '''
        src = self._source
        if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
            return src
        else:
            while True:
                src = src.source
                if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
                    return src

    def flatten(self, **kwargs):
        self._dset = self._source.flatten()
        return self._derived_dset(**kwargs)

    def get_domain_dset(self, idomain, **kwargs):
        self._dset = self._source.get_domain_dset(idomain)
        return self._derived_dset(**kwargs)

    def sample_points(self, pxyz, **kwargs):
        import pymses

        # source = self.source
        # if hasattr(source, "source"):
        #     source = source.source
        source = self.source
        self._dset = pymses.analysis.sample_points(source, pxyz, use_C_code=True)
        return self._derived_dset(**kwargs)

    def _derived_dset(self, **kwargs):
        print "Deriving dataset..."

        if self._dset is None:
            return {}
        derived_dset = {}
        dset = self._dset
        family = self._family.family

        if "dx" in self.required_fields:
            dset.add_scalars("dx", dset.get_sizes())
        if "pos" in self.required_fields:
            dset.add_vectors("pos", dset.points)

        # Deal with tracked fields / units
        tracked_fields = {}
        for f in self.required_fields:
            if seren3.in_tracked_field_registry(f):
                unit_key = seren3.get_tracked_field_info_key(f)
                unit_string = seren3.get_tracked_field_unit(f)
                pymses_unit = seren3.pymses_units(unit_string)

                val = dset[f] * self._family.info[unit_key].express(pymses_unit)
                val = SimArray(val, unit_string)
                tracked_fields[f] = val
            else:
                tracked_fields[f] = SimArray(dset[f])

        # Derive fields
        def _get_derived(field, temp):
            '''
            Recursively derive all the fields we need
            '''
            rules = [r for r in seren3.required_for_field(family, field)]
            for r in rules:
                if seren3.is_derived(family, r) and r not in dset.fields:
                    # Recursively derive required field
                    _get_derived(r, temp)

                elif r in tracked_fields:
                    temp[r] = tracked_fields[r]

            fn = seren3.get_derived_field(family, field)
            temp[field] = fn(self._family.base, temp, **kwargs)

        for f in self.requested_fields:
            if f in dset.fields:  # tracked field
                derived_dset[f] = tracked_fields[f]
            elif f not in derived_dset and seren3.is_derived(family, f):
                temp = {}
                for r in self.required_fields:
                    if r == 'pos':
                        temp["pos"] = dset.points

                    elif r == 'dx':
                        temp["dx"] = dset.get_sizes()

                    else:
                        temp[r] = dset[r]

                _get_derived(f, temp)
                derived_dset[f] = temp[f]
            else:
                raise Exception("Don't know what to do with non-tracked and non-derived field: %s", f)

        print "Done"
        return derived_dset