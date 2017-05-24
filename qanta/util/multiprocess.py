import sys
import time
from multiprocessing import Pool, Manager
from qanta.config import conf

def _multiprocess(worker, inputs, n_cores=conf['buzzer']['n_cores'], info='',
        multi=True):

    total_size = len(inputs)
    output = '\r[{0}] done: {1}/{2}'
    if multi:
        pool = Pool(n_cores)
        manager = Manager()
        queue = manager.Queue()
        while not queue.empty():
            queue.get()
        inputs = [(i, queue) for i in inputs]
        result = pool.map_async(worker, inputs)
        # monitor loop
        while not result.ready():
            size = queue.qsize()
            sys.stderr.write(output.format(info, size, total_size))
            time.sleep(0.1)
        sys.stderr.write('\n')
        pool.close()
        return result.get()
    else:
        result = []
        inputs = [(i, None) for i in inputs]
        for i, inp in enumerate(inputs):
            result.append(worker(inp))
            sys.stderr.write(output.format(info, i, total_size))
        sys.stderr.write('\n')
        return result

