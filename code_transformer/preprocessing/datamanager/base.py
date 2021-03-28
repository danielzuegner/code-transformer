"""
Utility methods and classes for data loading.
The BufferedDataManager allows preloading of dataset slices in a separate thread.
"""

import collections
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread, Event

import numpy as np

from code_transformer.utils.log import get_logger
from code_transformer.utils.timing import Timing

logger = get_logger(__name__)


class RawDataLoader(ABC):

    @abstractmethod
    def read(self, batch_size=1, shuffle=False):
        pass

    @abstractmethod
    def __len__(self):
        pass


class DataManager(ABC):

    @staticmethod
    def to_batches(generator, batch_size, lazy=False):
        """
        Lazyly evaluated batch-wise loading of the code snippets
        """

        if batch_size == 1:
            for item in generator:
                yield item
            return

        if lazy:
            # Lazy returns batches as a generator where objects are only touched upon actually querying them
            iterator = iter(generator)
            try:
                while True:
                    first = next(iterator)

                    def chunk():
                        try:
                            yield first
                            for _ in range(batch_size - 1):
                                yield next(iterator)
                        except StopIteration:
                            pass

                    yield chunk()
            except StopIteration:
                pass
        else:
            # Regular mode materializes all objects within a batch before the batch is returned as a list
            batch = []
            for i, item in enumerate(generator):
                batch.append(item)
                if (i + 1) % batch_size == 0:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def read(self, batch_size=1):
        return self.to_batches(self, batch_size)

    @abstractmethod
    def save(self, data, **kwargs):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class BufferedDataManager(DataManager):
    """
    Wrapper class for arbitrary data managers that preloads samples in the background and provides asynchroneous saving.
    Useful in multiprocessing settings where we can have the main process preloading data while other processes do the
    work.
    The idea is to eliminate any waiting when iterating over the samples or when saving a dataset slice. This is
    obtained by using background worker threads that operate on queues instead of directly using the data_manager.
    To ensure that the python process can end after using a BufferedDataManager, one should call the .shutdown() method
    """

    QUEUE_END_MSG = 'DONE'  # A special message that is used for internal queues to signalize that the producer thread is done

    def __init__(self, data_manager: DataManager, size_load_buffer=5000, size_save_buffer=1):
        """
        :param data_manager: can be an arbitrary data manager that supports iterating over samples and saving dataset files
        :param size_load_buffer: specifies how many SAMPLES will be prefetched from data_manager
        :param size_save_buffer: specifies how many DATASET SLICES will be buffered until a call to .save() will actually block
        """

        self.data_manager = data_manager
        self.load_buffer = Queue(size_load_buffer)
        self.save_buffer = Queue(size_save_buffer)
        self.load_worker = None  # Will be initialized upon obtaining an iterator
        self.save_worker = None  # Will be initialized when the first file needs to be saved
        self.stop_event = Event()

    def __iter__(self):
        """
        Initializes a worker for prefetching data. The worker will start populating the internal queue once an iterator
        is created. To avoid spawning multiple workers, one can only have one iterator at a time.
        """

        if self.load_worker is not None:
            raise Exception("There is already an iterator running!")
        self.load_worker = self.LoadWorker(self.data_manager, self.load_buffer, self.stop_event)
        self.load_worker.start()
        return self

    def __next__(self):
        """
        Reads from the internal buffer and only blocks when it is empty. In this case, it might help to increase the
        size of the internal buffer via size_load_buffer
        """

        data = self.load_buffer.get()
        if data == self.QUEUE_END_MSG:
            # the load worker will put a special DONE MESSAGE to the internal queue to signal that the data_manager
            # won't provide more samples
            self.load_worker.join()
            self.load_worker = None
            raise StopIteration
        return data

    def __del__(self):
        """
        Destructor. Attempts to join all threads to allow the python script to exit cleanly.
        """

        self.shutdown()

    def save(self, data, **kwargs):
        """
        Puts the data on the internal save buffer and immediately returns. Only blocks when the internal save buffer
        is already full, i.e., the worker takes too longer to save one dataset slice than new data is incoming.
        In this case, this code has to be extended to allow for multiple save workers.
        A save worker is created when .save() is called for the first time.
        :param data: the data to be saved
        """

        if not self.save_worker:
            self.save_worker = self.SaveWorker(self.data_manager, self.save_buffer)
            self.save_worker.start()
        self.save_buffer.put(data)

    def shutdown(self):
        """
        Clears all the buffers, terminates all workers and prepares the buffered data manager to be used again.
        Should be called when one is done with iterating over the samples to allow the python process to end.
        :return:
        """

        self.stop_event.set()  # Signalize the load worker to shutdown
        if self.load_worker:
            if self.load_worker.is_alive() and not self.load_buffer.empty():
                # In this case, the load worker is waiting to put something into the queue and thus cannot receive the
                # stop signal. Resolve by taking one element out of the read buffer
                self.load_buffer.get()
            self.load_worker.join()

        # Possibly awake blocking SaveWorker and signalize that no more data
        # will be put to the save buffer, i.e., the worker can shutdown
        self.save_buffer.put(self.QUEUE_END_MSG)
        if self.save_worker:
            self.save_worker.join()

        self.load_buffer.queue.clear()
        self.save_buffer.queue.clear()
        self.stop_event = Event()
        self.load_worker = None
        self.save_worker = None

    class LoadWorker(Thread):
        """
        Background thread that iterates over all samples in data_manager and puts them onto the internal load buffer.
        To avoid out of memory issues, the internal queue has limited size which can be controlled via size_load_buffer
        """

        def __init__(self, data_manager, read_buffer, stop_event):
            Thread.__init__(self)
            self.data_manager = data_manager
            self.read_buffer = read_buffer
            self.stop_event = stop_event

        def run(self) -> None:
            for sample in self.data_manager.read():
                if self.stop_event.is_set():
                    return
                self.read_buffer.put(sample)
            self.read_buffer.put(BufferedDataManager.QUEUE_END_MSG)  # Signalize that the data_manager iterator is empty

    class SaveWorker(Thread):
        """
        Background thread that waits for data to be saved on the internal save buffer.
        Will run until a special DONE MESSAGE is put onto the queue.
        """

        def __init__(self, data_manager, save_buffer):
            Thread.__init__(self)
            self.data_manager = data_manager
            self.save_buffer = save_buffer

        def run(self) -> None:
            while True:
                data = self.save_buffer.get()
                if data == BufferedDataManager.QUEUE_END_MSG:
                    return
                with Timing() as t:
                    self.data_manager.save(data)
                logger.info(f"Saving {len(data)} samples took {t[0]:0.3f} seconds")


class CombinedDataManager(DataManager):

    def __init__(self, data_managers, identifiers=None):
        self.data_managers = data_managers
        self.identifiers = identifiers
        if identifiers is not None:
            assert len(data_managers) == len(
                identifiers), "Need to have the same amount of identifiers as data managers"

    def save(self, data, **kwargs):
        raise Exception('CombinedDataManager cannot save')

    def __iter__(self):
        return CombinedDataManager.Iterator([iter(data_manager) for data_manager in self.data_managers],
                                            self.identifiers)

    class Iterator:

        def __init__(self, iterators, identifiers):
            self.iterators = iterators
            self.identifiers = identifiers

        def __next__(self):
            if len(self.iterators) == 0:
                raise StopIteration()
            random_iterator_idx = np.random.choice(range(len(self.iterators)))
            try:
                sample = next(self.iterators[random_iterator_idx])
                if self.identifiers is not None:
                    return self.identifiers[random_iterator_idx], sample
                else:
                    return sample
            except StopIteration:
                del self.iterators[random_iterator_idx]
                if self.identifiers is not None:
                    del self.identifiers[random_iterator_idx]
                return next(self)


class DataLoaderWrapper(DataManager):

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return iter(self.dataloader)

    def save(self, data, **kwargs):
        pass


CTBatch = collections.namedtuple("CTBatch", ['tokens', 'token_types', 'node_types', 'relative_distances',
                                                         'distance_names', 'sequence_lengths', 'pad_mask', 'labels',
                                                         'perm_mask', 'target_mapping', 'target_mapping_per_token',
                                                         'extended_vocabulary',
                                                         'extended_vocabulary_ids', 'pointer_pad_mask', 'languages'])


def batch_to_device(batch: CTBatch, device="cuda"):
    replace_dict = {}
    for ix, k in enumerate(batch._fields):
        if k == "distance_names":
            replace_dict[k]: batch[ix]
        elif k == "relative_distances" and batch[ix] is not None:
            replace_dict[k] = [(x[0].to(device), x[1].to(device)) for x in batch[ix]]
        elif batch[ix] is not None and not isinstance(batch[ix], list):
            replace_dict[k] = batch[ix].to(device)
        else:
            replace_dict[k] = batch[ix]
    batch = batch._replace(**replace_dict)

    return batch


def batch_filter_distances(batch: CTBatch, distance_names):
    if not hasattr(batch, "distance_names"):
        # Filtering distances does not apply here
        return batch

    assert all([x in batch.distance_names for x in distance_names])
    ixs_keep = [ix for ix, x in enumerate(batch.distance_names) if x in distance_names]
    relative_distances = [x for ix, x in enumerate(batch.relative_distances) if ix in ixs_keep]
    new_distance_names = [x for ix, x in enumerate(batch.distance_names) if ix in ixs_keep]
    batch = batch._replace(distance_names=new_distance_names, relative_distances=relative_distances)
    return batch
