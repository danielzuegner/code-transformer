from contextlib import contextmanager
from time import time


@contextmanager
def timing(descr: str):
    start = time()
    yield
    print(f"{descr}: {time() - start:0.2f} seconds")


class Timing:
    """
    Usage:

    with Timing() as t:
        ...
    print(f"took {t[0]:0.2} seconds")

    Within the with-statement we can take several measurements by calling t.measure().
    Every measurement will be available after the closure in the list t.
    """

    class TimeMeasurer:

        def __init__(self, start):
            self.start = start
            self.times = []

        def measure(self):
            now = time()
            measured_time = now - self.start
            self.times.append(measured_time)
            self.start = now
            return measured_time

        def __len__(self):
            return len(self.times)

        def __getitem__(self, i):
            return self.times[i]

        def __str__(self):
            if len(self.times) == 1:
                return f"{self.times[0]:.2f}"
            return f"{self.times}"

    def __enter__(self):
        self.time_measurer = Timing.TimeMeasurer(time())
        return self.time_measurer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_measurer.measure()
