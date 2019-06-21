"""
Utility functions to include drift velocity in grafIC ics by computing/convolving density
power spectrum k dependent bias. Contains routines to run CICsASS
"""
import numpy as np

def fft_sample_spacing(N, boxsize):
    from seren3.cosmology import _power_spectrum
    return _power_spectrum.fft_sample_spacing(N, boxsize)


def fft_sample_spacing_components(N):
    from seren3.cosmology import _power_spectrum
    return _power_spectrum.fft_sample_spacing_components(N)

def vbc_rms(vbc_field):
    '''
    Computes the rms vbc in the box
    '''
    rms = np.sqrt(np.mean(vbc_field ** 2))
    return rms


def vbc_ps_fname(rms, z, boxsize):
    import os
    cwd = os.getcwd()
    if not os.path.isdir("%s/vbc_TFs_out" % cwd):
        os.mkdir("%s/vbc_TFs_out" % cwd)
    return '%s/vbc_TFs_out/vbc_%f_z%f_B%1.2f.dat' % (cwd, rms, z, boxsize)


def run_cicsass_lc(boxsize, z, rms_vbc_z1000, out_fname, N=256):
    import subprocess, os
    from seren3.utils import which

    exe = which('transfer.x')

    if exe is None:
        raise Exception("Unable to locate transfer.x executable")

    # Example execution for RMS vbc=30km/s @ z=1000.:
    # ./transfer.x -B0.2 -N128 -V30 -Z100 -D3 -SinitSB_transfer_out

    CICsASS_home = os.getenv("CICSASS_HOME")
    if CICsASS_home is None:
        raise Exception("Env var CICSASS_HOME not set")

    # Run with N=256
    # CICsASS_home = "/lustre/scratch/astro/ds381/CICsASS/matt/Dropbox/CICASS/vbc_transfer/"
    cmd = 'cd %s && %s -B%1.2f -N%d -V%f -Z%f -D3 -Splanck2018_transfer_out > %s' % (
        CICsASS_home, exe, boxsize, N, rms_vbc_z1000, z, out_fname)
    # print 'Running:\n%s' % cmd

    # Run CICsASS and wait for output, if it doesn't work properly
    # then it will raise a CalledProcessError
    try:
        output = subprocess.check_output(cmd, shell=True)
    except CalledProcessError:
        raise Exception("CICsASS returned non-zero exit code: %d", code)
    
    output = output.decode("ascii")
    output = output.splitlines()
    
    vals = np.zeros(shape=(64, 4))

    for i in range(64):
        vals[i, :] = output[i].split()

    return vals


def run_cicsass(boxsize, z, rms_vbc_z1000, out_fname, N=256):
    import subprocess, os
    from seren3.utils import which

    exe = which('transfer.x')

    if exe is None:
        raise Exception("Unable to locate transfer.x executable")

    # Example execution for RMS vbc=30km/s @ z=1000.:
    # ./transfer.x -B0.2 -N128 -V30 -Z100 -D3 -SinitSB_transfer_out

    CICsASS_home = os.getenv("CICSASS_HOME")
    if CICsASS_home is None:
        raise Exception("Env var CICSASS_HOME not set")

    # Run with N=256
    # CICsASS_home = "/lustre/scratch/astro/ds381/CICsASS/matt/Dropbox/CICASS/vbc_transfer/"
    cmd = 'cd %s && %s -B%1.2f -N%d -V%f -Z%f -D3 -Splanck2018_transfer_out > %s' % (
        CICsASS_home, exe, boxsize, N, rms_vbc_z1000, z, out_fname)
    # print 'Running:\n%s' % cmd
    # Run CICsASS and wait for output
    code = subprocess.check_call(cmd, shell=True)
    if code != 0:
        raise Exception("CICsASS returned non-zero exit code: %d", code)
    return code


def compute_velocity_bias(ics, vbc):
    import os, time
    from seren3.array import SimArray
    # print 'AVERAGE INSTEAD OF RMS'
    # Init fields
    if vbc is None:
        vbc = ics['vbc']

    # Compute size of grid and boxsize
    N = vbc.shape[0]
    boxsize = float(ics.boxsize) * \
        (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    # vbc_norm = ics.vbc_rms_norm(vbc=vbc)
    # vbc_rms = vbc_norm * (1001.)  # vbc_rms prop (1 + z)
    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

    # Check for PS and run CICsASS if necessary
    fname_vbc0 = vbc_ps_fname(0., z, boxsize)
    if os.path.isfile(fname_vbc0) is False:
        exit_code = run_cicsass(boxsize, z, 0., fname_vbc0)

    fname_vbcrecom = vbc_ps_fname(rms_recom, z, boxsize)
    if os.path.isfile(fname_vbcrecom) is False:
        exit_code = run_cicsass(boxsize, z, rms_recom, fname_vbcrecom)

    # Load the power spectra and compute the bias
    # LC - might be too quick for CICASS, check for empty files
    ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
    ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)
    count = 0
    while ((len(ps_vbc0) == 0) or (len(ps_vbcrecom) == 0)):
        count += 1
        if count > 10:
            raise Exception("Reached sleep limit. File still empty.")
            print("Caught exception (fname_vbc0): {0}".format(fname_vbc0))
            print("Caught exception (fname_vbcrecom): {0}".format(fname_vbcrecom))
        time.sleep(5)
        ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
        ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)
    
    # Should have same lenghts if finished writing
    count = 0
    try:
        while len(ps_vbcrecom[1]) != len(ps_vbc0[1]):
            count += 1
            if count > 10:
                raise Exception("Reached sleep limit. Filesizes still differ.")
            time.sleep(5)
            ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
            ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)
    except Exception as e:
        print("Caught exception (fname_vbc0): {0}".format(fname_vbc0))
        print("Caught exception (fname_vbcrecom): {0}".format(fname_vbcrecom))

    cosmo = ics.cosmo

    from seren3 import cosmology
    vdeltab0 = cosmology.linear_velocity_ps(
        ps_vbc0[0], np.sqrt(ps_vbc0[2]), **cosmo)
    vdeltab = cosmology.linear_velocity_ps(
        ps_vbcrecom[0], np.sqrt(ps_vbcrecom[2]), **cosmo)

    vdeltac0 = cosmology.linear_velocity_ps(
        ps_vbc0[0], np.sqrt(ps_vbc0[1]), **cosmo)
    vdeltac = cosmology.linear_velocity_ps(
        ps_vbcrecom[0], np.sqrt(ps_vbcrecom[1]), **cosmo)

    #CDM bias
    b_cdm = vdeltac / vdeltac0
    # Baryon bias/p/scratch/chpo22/hpo22i/bd/cicass/vbc_transfer/vbc_TFs_out/vbc_22.435140_z200.000005_B3.52.dat
    b_b = vdeltab / vdeltab0
    # Wavenumber
    k_bias = SimArray(ps_vbcrecom[0] / ics.cosmo["h"], "h Mpc**-1")

    return k_bias, b_cdm, b_b


def compute_velocity_bias_lc(ics, vbc):
    import os, time
    from seren3.array import SimArray
    # print 'AVERAGE INSTEAD OF RMS'
    # Init fields
    if vbc is None:
        vbc = ics['vbc']

    # Compute size of grid and boxsize
    N = vbc.shape[0]
    boxsize = float(ics.boxsize) * \
        (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    # vbc_norm = ics.vbc_rms_norm(vbc=vbc)
    # vbc_rms = vbc_norm * (1001.)  # vbc_rms prop (1 + z)
    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

    ps_vbc0 = run_cicsass_lc(boxsize, z, 0., fname_vbc0)
    ps_vbcrecom = run_cicsass_lc(boxsize, z, rms_recom, fname_vbcrecom)

    cosmo = ics.cosmo

    from seren3 import cosmology
    vdeltab0 = cosmology.linear_velocity_ps(
        ps_vbc0[0], np.sqrt(ps_vbc0[2]), **cosmo)
    vdeltab = cosmology.linear_velocity_ps(
        ps_vbcrecom[0], np.sqrt(ps_vbcrecom[2]), **cosmo)

    vdeltac0 = cosmology.linear_velocity_ps(
        ps_vbc0[0], np.sqrt(ps_vbc0[1]), **cosmo)
    vdeltac = cosmology.linear_velocity_ps(
        ps_vbcrecom[0], np.sqrt(ps_vbcrecom[1]), **cosmo)

    #CDM bias
    b_cdm = vdeltac / vdeltac0
    # Baryon bias/p/scratch/chpo22/hpo22i/bd/cicass/vbc_transfer/vbc_TFs_out/vbc_22.435140_z200.000005_B3.52.dat
    b_b = vdeltab / vdeltab0
    # Wavenumber
    k_bias = SimArray(ps_vbcrecom[0] / ics.cosmo["h"], "h Mpc**-1")

    return k_bias, b_cdm, b_b


def compute_cicsass(ics, vbc):
    """Function used to calculate all the cicass power spectra before
    doing anything else. Not very efficient, but might be necessary."""
    import os, time
    from seren3.array import SimArray
   
    # Compute size of grid and boxsize (for this patch)
    N = vbc.shape[0]
    boxsize = ics.boxsize.in_units("Mpc a h**-1") * (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

        # Check for PS and run CICsASS if needed
    fname_vbc0 = vbc_ps_fname(0., z, boxsize)
    if not os.path.isfile(fname_vbc0):
        exit_code = run_cicsass(boxsize, z, 0., fname_vbc0)

    fname_vbcrecom = vbc_ps_fname(rms_recom, z, boxsize)
    if not os.path.isfile(fname_vbcrecom):
        exit_code = run_cicsass(boxsize, z, rms_recom, fname_vbcrecom)


def compute_bias(ics, vbc):
    """ Calculate the bias to the density power spectrum assuming
    COHERENT vbc at z=1000. """
    import os, time
    from seren3.array import SimArray
   
    # Compute size of grid and boxsize (for this patch)
    N = vbc.shape[0]
    boxsize = ics.boxsize.in_units("Mpc a h**-1") * (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

    # Check for PS and run CICsASS if needed
    fname_vbc0 = vbc_ps_fname(0., z, boxsize)
    if not os.path.isfile(fname_vbc0):
        exit_code = run_cicsass(boxsize, z, 0., fname_vbc0)

    fname_vbcrecom = vbc_ps_fname(rms_recom, z, boxsize)
    if not os.path.isfile(fname_vbcrecom):
        exit_code = run_cicsass(boxsize, z, rms_recom, fname_vbcrecom)

    # Load the power spectra and compute the bias
    # LC - might be too quick for CICASS, check for empty files
    ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
    ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)
    count = 0
    while ((len(ps_vbc0) == 0) or (len(ps_vbcrecom) == 0)):
        count += 1
        if count > 10:
            raise Exception("Reached sleep limit. File still empty.")
            print("Caught exception (fname_vbc0): {0}".format(fname_vbc0))
            print("Caught exception (fname_vbcrecom): {0}".format(fname_vbcrecom))
        time.sleep(5)
        ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
        ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)

    # Should have same lenghts if finished writing
    count = 0
    try:
        while len(ps_vbcrecom[1]) != len(ps_vbc0[1]):
            count += 1
            if count > 10:
                raise Exception("Reached sleep limit. Filesizes still differ")
            time.sleep(5)
            ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
            ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)
    except Exception as e:
        print("Caught exception (fname_vbc0): {0}".format(fname_vbc0))
        print("Caught exception (fname_vbcrecom): {0}".format(fname_vbcrecom))

    #CDM bias
    b_cdm = ps_vbcrecom[1] / ps_vbc0[1]
    # Baryon bias
    b_b = ps_vbcrecom[2] / ps_vbc0[2]
    # Wavenumber
    k_bias = SimArray(ps_vbcrecom[0] / ics.cosmo["h"], "h Mpc**-1")

    return k_bias, b_cdm, b_b


def compute_bias_lc(ics, vbc):
    """ Calculate the bias to the density power spectrum assuming
    COHERENT vbc at z=1000. """
    import os, time
    from seren3.array import SimArray
   
    # Compute size of grid and boxsize (for this patch)
    N = vbc.shape[0]
    boxsize = ics.boxsize.in_units("Mpc a h**-1") * (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

    ps_vbc0 = run_cicsass_lc(boxsize, z, 0., fname_vbc0)
    ps_vbcrecom = run_cicsass_lc(boxsize, z, rms_recom, fname_vbcrecom)

    #CDM bias
    b_cdm = ps_vbcrecom[1] / ps_vbc0[1]
    # Baryon bias
    b_b = ps_vbcrecom[2] / ps_vbc0[2]
    # Wavenumber
    k_bias = SimArray(ps_vbcrecom[0] / ics.cosmo["h"], "h Mpc**-1")

    return k_bias, b_cdm, b_b


def apply_density_bias(ics, k_bias, b, N, delta_x=None):
    ''' Apply a bias to the realisations power spectrum, and recompute the 3D field.
    Parameters:
        b (array): bias to deconvolve with the delta_x field, such that:
        delta_x = ifft(delta_k/b)
    '''
    import scipy.fftpack as fft
    import scipy.interpolate as si

    if delta_x is None:
        delta_x = ics['deltab']

    shape = delta_x.shape

    boxsize = float(ics.boxsize) * \
        (float(N) / float(ics.header.N))

    # print "boxsize = ", boxsize, delta_x.shape[0]

    k = None
    if boxsize != ics.boxsize:
        # Resample k as we may be using a subregion
        k = fft_sample_spacing(delta_x.shape[0], boxsize).flatten()
    else:
        k = ics.k.flatten()
    k[k == 0.] = (2. * np.pi) / boxsize

    # Interpolate/extrapolate the bias to the 3D grid
    def log_interp1d(xx, yy, kind='linear'):
        logx = np.log10(xx)
        logy = np.log10(yy)
        lin_interp = si.InterpolatedUnivariateSpline(logx, logy)
        log_interp = lambda zz: np.power(
            10.0, lin_interp(np.log10(zz)))
        return log_interp
    f = log_interp1d(k_bias, b)
    b = f(k)

    delta_k = fft.fftn(delta_x)

    # Apply the bias
    delta_k *= np.sqrt(b.reshape(delta_k.shape))

    # Inverse FFT to compute the realisation
    delta_x = fft.ifftn(delta_k).real.reshape(shape)
    return delta_x

def cube_positions(ics, n, N=None):
    cubes = []
    if N is None:
        N = ics.header.N

    if (N % n != 0):
        raise Exception(
            "Cannot fit %d cubes into grid with size %d" % (n, self.N))

    dx_cells = N / n

    for i in range(n):
        cen_i = dx_cells * (i + 0.5)

        for j in range(n):
            cen_j = dx_cells * (j + 0.5)

            for k in range(n):
                cen_k = dx_cells * (k + 0.5)
                cubes.append([cen_i, cen_j, cen_k])

    return cubes, dx_cells
