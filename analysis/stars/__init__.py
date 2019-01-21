def test_sfr():
    '''
    Test SFR calculation is working correctly
    '''
    import seren3
    import numpy as np
    from scipy import integrate

    path = "/research/prace/david/aton/256/"
    sim = seren3.init(path)
    iout = sim.numbered_outputs[-1]  # last snapshot

    snap = sim[iout]
    snap_sfr, lbtime, bsize = sfr(snap)

    dset = snap.s["mass"].flatten()
    mstar_tot = dset["mass"].in_units("Msol").sum()
    integrated_mstar = integrate.trapz(snap_sfr, lbtime)

    assert(np.allclose(mstar_tot, integrated_mstar, rtol=1e-2)), "Error: Integrated stellar mass not close to actual."
    print "Passed"

def sfr(context, dset=None, ret_sSFR=False, nbins=100, **kwargs):
    '''
    Compute the (specific) star formation rate within this context.
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.exceptions import NoParticlesException

    if (dset is None):
        dset = context.s[["age", "mass"]].flatten()
    age = dset["age"].in_units("Gyr")
    mass = dset["mass"].in_units("Msol")

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'analysis/stars/sfr')

    def compute_sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "yr**-1")

        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return SimArray(sfrhist, "Msol yr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")

    sfrhist, lookback_time, binsize = compute_sfr(age, mass, **kwargs)
    SFR = sfrhist.in_units("Msol Gyr**-1")

    SFR.set_field_latex("$\\mathrm{SFR}$")
    lookback_time.set_field_latex("$\\mathrm{Lookback-Time}$")
    binsize.set_field_latex("$\Delta$")

    if ret_sSFR:
        sSFR = SFR / mass.sum()  # specific star formation rate
        return sSFR, SFR, lookback_time, binsize
    return SFR, lookback_time, binsize  # SFR [Msol Gyr^-1] (sSFR [Gyr^-1]), Lookback Time [Gyr], binsize [Gyr]

def gas_SFR_density(context, impose_criterion=True, return_averages=False):
    '''
    Computes the (instantaneous) star formation rate density, in Msun/yr/kpc^3, from the gas
    '''
    import numpy as np
    from seren3.core.namelist import NML

    if hasattr(context, "subsnap"):
        # Halos
        context = context.subsnap

    mH = context.array(context.C.mH)
    X_fraction = context.info.get("X_fraction", 0.76)
    H_frac = mH/X_fraction  # fractional mass of hydrogen

    nml = context.nml
    dset = None
    if (return_averages):
        dset = context.g[["nH", "T2", "mass", "dx"]].flatten()
    else:
        dset = context.g[["nH", "T2"]].flatten()

    nH = dset["nH"].in_units("cm**-3")

    # Load star formation model params from the namelist

    # First, check whether using legacy (PHYSICS_PARAMS) or new
    # (SF_PARAMS) namelist blocks
    
    if NML.PHYSICS_PARAMS in nml:
        print("Using legacy namelist block (PHYSICS_PARAMS)")
        n_star = context.array(nml[NML.PHYSICS_PARAMS]["n_star"], "cm**-3")  # cm^-3
        t_star = context.quantities.t_star.in_units("yr")

    elif NML.SF_PARAMS in nml:
        n_star = context.array(nml[NML.SF_PARAMS]["n_star"], "cm**-3")  # cm^-3
        t_star = context.quantities.t_star.in_units("yr")

    # Compute the SFR density in each cell
    sfr = nH / (t_star*np.sqrt(n_star/nH))  # atoms/yr/cm**3
    sfr *= H_frac  # kg/yr/cm**3
    sfr.convert_units("Msol yr**-1 kpc**-3")

    # Impose density/temperature criterion
    if impose_criterion:
        idx = np.where(nH < n_star)
        sfr[idx] = 0.

        # Compute and subtract away the non-thermal polytropic temperature floor

        # First, check whether using a namelist with legacy
        # (PHYSICS_PARAMS) or new (SF_PARAMS) namelist blocks
        if NML.PHYSICS_PARAMS in nml:
            print("Using legacy namelist block (PHYSICS_PARAMS)")
            g_star = nml[NML.PHYSICS_PARAMS].get("g_star", 1.)
            T2_star = context.array(nml[NML.PHYSICS_PARAMS]["T2_star"], "K")
        elif NML.SF_PARAMS in nml:
            g_star = nml[NML.SF_PARAMS].get("g_star", 1.)
            T2_star = context.array(nml[NML.SF_PARAMS]["T2_star"], "K")
            
        Tpoly = T2_star * (nH/n_star)**(g_star-1.)  # Polytropic temp. floor
        Tmu = dset["T2"] - Tpoly  # Remove non-thermal polytropic temperature floor
        idx = np.where(Tmu > 2e4)
        sfr[idx] = 0.

    if (return_averages):
        from seren3.analysis import volume_mass_weighted_average
        vw, mw = volume_mass_weighted_average(context, sfr, dset)
        return sfr, vw, mw

    return sfr 

