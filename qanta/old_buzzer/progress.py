import datetime
import sys
import time
from time import gmtime, strftime

class ProgressBar(object):

    def __init__(self, length, unit_iteration=False, update_interval=1,
            bar_length=40, out=sys.stdout):
        self._length = length 
        self._unit_iteration = unit_iteration
        self._update_interval = update_interval
        self._bar_length = bar_length
        self._out = out
        self._recent_timing = []

    def __call__(self, iteration, epoch):
        length = self._length
        unit_iteration = self._unit_iteration
        out = self._out

        if iteration % self._update_interval == 0:
            recent_timing = self._recent_timing
            now = time.time()
            
            recent_timing.append((iteration, epoch, now))

            out.write('\033[J')

            rate = iteration / length if unit_iteration else epoch / length

            bar_length = self._bar_length
            marks = '#' * int(rate * bar_length)
            out.write('         total [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), rate))

            epoch_rate = epoch - int(epoch)
            marks = '#' * int(epoch_rate * bar_length)
            out.write('    this epoch [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), epoch_rate))

            out.write("{:10} iter, {} epochs / {} \n".format(
                iteration, int(epoch), length))

            old_t, old_e, old_sec = recent_timing[0]
            span = now - old_sec
            if span != 0:
                speed_t = (iteration - old_t) / span
                speed_e = (epoch - old_e) / span
            else:
                speed_t = float('inf')
                speed_e = float('inf')

            if unit_iteration:
                estimated_time = (length - iteration) / speed_t
            else:
                estimated_time = (length - epoch) / speed_e

            out.write('{:10.5g} iters/sec. Estimated time to finish: {}.\n'.format(
                    speed_t, datetime.timedelta(seconds=estimated_time)))

            # move the cursor to the head of the progress bar
            out.write('\033[4A')
            out.flush()

            if len(recent_timing) > 100:
                del recent_timing[0]

    def finalize(self):
        # delete the progress bar
        out = self._out
        out.write('\033[J')
        out.flush()
