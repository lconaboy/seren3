import numpy as np
import os

################################################################################
# LC - profiling memory usage
import psutil
def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
################################################################################

# VERBOSE = 1  # 0 for all, >0 for just patch, <0 for none
P = False
B = False
C = False

class Result(object):
    '''
    Simple wrapper object to contain result of single iteration MPI computation
    '''
    def __init__(self, rank, idx):
        self.rank = rank
        self.idx = idx
        self.result = None

    def __repr__(self):
        return "rank: %d idx: %s result: %s" % (self.rank, self.idx, self.result)

    def __eq__(self, other):
        return self.result == other.result

    def __ne__(self, other):
        return self.result != other.result

    def __hash__(self):
        return hash(self.result)

    
class Patch(object):

    def __init__(self, patch, dx, field):
        self.patch = patch
        self.dx = dx
        self.field = field



def work(patch, dx, rank):
    ##########################################################
    # LC - profile memory usage
    print("rank {0}, memory usage = {1:.3f} Mo".format(rank, get_memory_usage()))
    ##########################################################
    origin = np.array(patch - float(dx) / 2. - pad, dtype=np.int64)
    dx_eps = float(dx) + float(2 * pad)

    delta = vbc = None
    if (P): print("Loading patch: %s" % patch)
    delta = ics.lazy_load_periodic("deltab", origin, int(dx_eps))
    vbc = ics.lazy_load_periodic("vbc", origin, int(dx_eps))

    # Compute the bias
    if (B): print("Computing bias")
    k_bias, b_cdm, b_b = vbc_utils.compute_bias_lc(ics, vbc)

    # Convolve with field
    if (C): print("Performing convolution")
    delta_biased = vbc_utils.apply_density_bias(ics, k_bias, b_b, delta.shape[0], delta_x=delta)

    # Remove the padded region
    x_shape, y_shape, z_shape = delta_biased.shape
    delta_biased = delta_biased[0 + pad:x_shape - pad,
                                0 + pad:y_shape - pad, 0 + pad:z_shape - pad]

    # Store
    biased_patch = Patch(patch, dx, delta_biased)

    return biased_patch


def main(path, level, patch_size):
    '''
    Writes a new set of grafIC initial conditions with a drift velocity dependent
    bias in the power spectrum
    '''
    from seren3.core import grafic_snapshot
    from seren3.analysis import drift_velocity as vbc_utils
    #from seren3.analysis.parallel import mpi
    from mpi4py import MPI
    from seren3.utils import divisors
    import gc

    # MPI stuff
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    #mpi.msg("Loading initial conditions")
    print("Loading initial conditions")
    
    ics = grafic_snapshot.load_snapshot(path, level, sample_fft_spacing=False)

    if mpi.host:
        # Make sure vbc field exists on disk
        if not ics.field_exists_on_disk("vbc"):
            ics.write_field(ics["vbc"], "vbc")

    div = np.array([float(i) for i in divisors(ics.header.N, mode='yield')])
    idx = np.abs((ics.header.N / div) * ics.dx - patch_size).argmin()
    ncubes = int(div[idx])

    # Compute cube positions in cell units
    cubes = dx = None
    if rank == 0:
        print("Using %i cubes per dimension." % ncubes)
        cubes, dx = vbc_utils.cube_positions(ics, ncubes)
        cubes = np.array(cubes)
        # Split the cubes into chunks that can be scattered to each processor 
        chunks = np.array_split(cubes, size)
    
    # dx is the same for every cube, so this is broadcast to each processor
    dx = comm.bcast(dx, root=0)
    pad = 8


############################## WORK LOOP ######################################

    # Iterate over patch positions in parallel
    # dest = {}
    patches = comm.scatter(chunks, root=0)
    biased_patches = [work(patch, dx, rank) for patch in patches]
    results = comm.gather(biased_patches, rank=0)

############################## END OF WORK LOOP ###############################
    print ('Done!')
    # if mpi.host:
    #     import os
    #     # Write new ICs
    #     dest = mpi.unpack(dest)

    #     output_field = np.zeros(ics.header.nn)

    #     for item in dest:
    #         result = item.result
    #         patch = result["patch"]
    #         dx = result["dx"]
    #         delta_biased = result["field"]

    #         # Bounds of this patch
    #         x_min, x_max = (int((patch[0]) - (dx / 2.)), int((patch[0]) + (dx / 2.)))
    #         y_min, y_max = (int((patch[1]) - (dx / 2.)), int((patch[1]) + (dx / 2.)))
    #         z_min, z_max = (int((patch[2]) - (dx / 2.)), int((patch[2]) + (dx / 2.)))

    #         # Place into output
    #         output_field[x_min:x_max, y_min:y_max, z_min:z_max] = delta_biased

    #     # Write the initial conditions
    #     ics_dir = "%s/ics_ramses_vbc/" % ics.level_dir
    #     if not os.path.isdir(ics_dir):
    #         os.mkdir(ics_dir)
    #     out_dir = "%s/level_%03i/" % (ics_dir, level)
    #     if not os.path.isdir(out_dir):
    #         os.mkdir(out_dir)

    #     ics.write_field(output_field, "deltab", out_dir=out_dir)

if __name__ == "__main__":
    import sys
    import traceback
    
    path = sys.argv[1]
    level = int(sys.argv[2])
    patch_size = float(sys.argv[3])

    #try:
    main(path, level, patch_size)
    # except Exception as e:
    #     from seren3.analysis.parallel import mpi
    #     mpi.msg("Caught exception (message): %s" % e.message)
    #     mpi.msg(traceback.format_exc())
    #     mpi.terminate(500, e=e)

    print("Done!")
