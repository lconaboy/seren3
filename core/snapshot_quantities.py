'''
Snapshot level quantities that can be calculated, i.e volume/mass weighted averages
'''
import numpy as np
from seren3.array import SimArray

class Quantity(object):
    def __init__(self, snapshot):
        self.base = snapshot

    def rho_mean(self, species='baryon'):
        '''
        Mean density at current redshift of baryons or cdm
        '''
        from seren3.cosmology import rho_mean_z
        cosmo = self.base.cosmo
        omega_0 = 0.
        if (species == 'b') or (species == 'baryon'):
            omega_0 = cosmo['omega_b_0']
        elif (species == 'c') or (species == 'cdm'):
            omega_0 = cosmo['omega_M_0'] - cosmo['omega_b_0']
        else:
            raise Exception("Unknown species %s" % species)

        rho_mean = rho_mean_z(omega_0, **cosmo)
        return SimArray(rho_mean, "kg m**-3")

    def box_mass(self, species='baryon'):
        snap = self.base
        rho_mean = self.rho_mean(species)  # kg / m^3
        boxsize = snap.info['unit_length'].express(snap.C.m)
        mass = rho_mean * boxsize**3.  # kg
        return SimArray(mass, "kg")

    def age_of_universe_gyr(self):
        fr = self.base.friedmann
        age_simu = fr["age_simu"]
        return SimArray(age_simu, "Gyr")

    def volume_weighted_average(self, field, mem_opt = False):
        '''
        Computes the volume weighted ionization fraction for the desired field
        '''
        boxsize = SimArray(self.base.info["boxlen"], self.base.info["unit_length"]).in_units("pc")
        if mem_opt:  # slower
            vsum = 0.
            for dset in self.base.g[[field, 'dx']]:
                dx = dset["dx"].in_units("pc")
                vsum += np.sum(dset[field] * dx**3)

            return vsum / boxsize**3
        else:
            dset = self.base.g[[field, 'dx']].flatten()
            dx = dset["dx"].in_units("pc")
            return np.sum(dset[field] * dx**3) / boxsize**3

    def mass_weighted_average(self, field, mem_opt = False):
        '''
        Computes the mass weighted ionization fraction for the desired field
        '''
        snap = self.base
        boxmass = self.box_mass('b').in_units("Msol")
        if mem_opt:
            msum = 0.
            for dset in snap.g[[field, 'mass']]:
                msum += np.sum(dset[field] * dset['mass'])

            return (msum / boxmass)
        else:
            dset = snap.g[[field, 'mass']].flatten()
            return np.sum(dset[field] * dset['mass']) / boxmass