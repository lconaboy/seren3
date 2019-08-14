import numpy as np
import os

from collections import Mapping, Container
from sys import getsizeof

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

This is a recursive function that drills down a Python object graph
like a dictionary holding nested dictionaries with lists of lists
and tuples and sets.

The sys.getsizeof function does a shallow size of only. It counts each
object inside a container as pointer only regardless of how big it
really is.

:param o: the object
:param ids:
:return:
"""
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r


# Setup MPI environment
from mpi4py import MPI
# MPI runtime variables
HOST_RANK=0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
host = (rank == HOST_RANK)

################################################################################
# LC - profiling memory usage
import psutil
def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
################################################################################


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


def is_host():
    return host


def piter(iterable, storage=None, keep_None=False, print_stats=False):
    '''
    Embarrassingly parallel MPI for loop
    Chunks and scatters iterables before gathering the results at the end
    Parameters
    ----------
    iterable : np.ndarray
        List to parallelise over - must be compatible with pickle (i.e numpy arrays, required by mpi4py)
    storage = None : dict
        Dictionary to store final (reduced) result on rank 0, if desired. Must be an empty dictionary

    Example
    ----------
    test = np.linspace(1, 10, 10)

    dest = {}

    for i, sto in piter(test, storage=dest):
        sto.result = i**2

    # Access result in dest (keys are rank numbers, values are lists of Result objects)
    '''

    # Chunk the iterable and prepare to scatter
    if host:
        if not hasattr(iterable, "__iter__"):
            terminate(500, e=Exception("Argument %s is not iterable!" % iterable))

        if (storage is not None) and (not isinstance(storage, dict)):
            raise Exception("storage must be a dict")

    if host:
        chunks = np.array_split(iterable, size)
    else:
        chunks = None

    # Scatter
    local_iterable = comm.scatter(chunks, root=0)
    del chunks

    if print_stats:
        msg("Received %d items" % len(local_iterable))

    local_results = np.zeros(len(local_iterable), dtype=object)

    # yield the iterable
    for i in xrange(len(local_iterable)):

        ##########################################################
        # LC - profile memory usage
        print("rank {0}, memory usage = {1:.3f} Mo".format(rank, get_memory_usage()))
        ##########################################################

        if print_stats:
            msg("%i / %i" % (i, len(local_iterable)))

        # yield to the for loop
        if storage is not None:
            #init the result
            res = Result(rank, i)
            yield local_iterable[i], res

            if keep_None is False and res.result is None:
                # Dont append None result to list
                continue

            # appending is expensive, try preallocating
            local_results[i] = res
            print("rank {0}, res.size = {1:.3f} Mo".format(rank, deep_getsizeof(res, set())))
        
        else:
            yield local_iterable[i]

    # If the executing code sets storage, then reduce it to rank 0
    if storage is not None:
        if print_stats:
            msg("Pending gather")
        results = comm.gather(local_results, root=0)
        del local_results

        if host:
            # Append results to storage
            for irank in range(size):
                local_results = results[irank]
                storage[irank] = local_results


def unpack(dest):
    '''
    Flatten the dictionary from piter (keys are mpi rank)
    '''
    result = []
    for rank in dest:
        for item in dest[rank]:
            result.append(item)
    return result


def msg(message):
    print '[rank %d   ]: %s' % (rank, message)


def terminate(code, e=None):
    if e:
        msg("Caught exception: %s" % e)
    msg("Terminating")
    comm.Abort(code)

