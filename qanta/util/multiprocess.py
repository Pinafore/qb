import sys
import time
from multiprocessing import Pool, Manager
from functools import partial
from qanta.config import conf

def queue_wrapper(func, inputs):
    real_inputs, queue = inputs
    queue.put(0)
    return func(*real_inputs)

def _multiprocess(func, inputs, n_cores=conf['buzzer']['n_cores'], info='',
        multi=True):

    total_size = len(inputs)
    output = '\r[{0}] ({1}) done: {2}/{3}'
    if multi:
        pool = Pool(n_cores)
        manager = Manager()
        queue = manager.Queue()
        while not queue.empty():
            queue.get()
        worker = partial(queue_wrapper, func)
        inputs = [(i, queue) for i in inputs]
        result = pool.map_async(worker, inputs)
        # monitor loop
        while not result.ready():
            size = queue.qsize()
            sys.stderr.write(output.format(info, n_cores, size, total_size))
            time.sleep(0.1)
        sys.stderr.write('\n')
        pool.close()
        return result.get()
    else:
        result = []
        for i, inp in enumerate(inputs):
            result.append(func(*inp))
            sys.stderr.write(output.format(info, 1, i, total_size))
        sys.stderr.write('\n')
        return result

