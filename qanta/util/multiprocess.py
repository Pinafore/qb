import sys
import time
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
from qanta.config import conf
from qanta.spark import create_spark_context


def queue_wrapper(func, inputs):
    real_inputs, queue = inputs
    queue.put(0)
    return func(*real_inputs)
  

def _multiprocess(func, inputs, n_cores=0, info='', 
                  progress=True, multi=True, spark=None):
    if n_cores == 0:
        n_cores = multiprocessing.cpu_count()
    total_size = len(inputs)
    output = '\r[{0}] ({1}) done: {2}/{3} eta: {4}'
    if spark is not None:
        def spark_wrapper(inputs):
            return func(*inputs)
        return spark.parallelize(inputs, 64 * n_cores).map(spark_wrapper).collect()
    elif multi:
        pool = Pool(n_cores)
        manager = Manager()
        queue = manager.Queue()
        while not queue.empty():
            queue.get()
        worker = partial(queue_wrapper, func)
        inputs = [(i, queue) for i in inputs]
        result = pool.map_async(worker, inputs)
        # monitor loop

        start_time = time.time()
        while not result.ready():
            size = queue.qsize()
            if size > 0:
                eta = int((time.time() - start_time) / size \
                        * (total_size - size))
                eta = '{}min {}s'.format(eta // 60, eta % 60)
            else:
                eta = 'inf'
            if progress:
                sys.stderr.write(
                        output.format(info, n_cores, size, total_size, eta))
            time.sleep(0.1)
        if progress:
            sys.stderr.write('\n')
        pool.close()
        return result.get()
    else:
        result = []
        start_time = time.time()
        for i, inp in enumerate(inputs):
            result.append(func(*inp))
            size = i
            if size > 0:
                eta = int((time.time() - start_time) / size \
                        * (total_size - size))
                eta = '{}min {}s'.format(eta // 60, eta % 60)
            else:
                eta = 'inf'
            if progress:
                sys.stderr.write(output.format(info, 1, i, total_size, eta))
        if progress:
            sys.stderr.write('\n')
        return result

