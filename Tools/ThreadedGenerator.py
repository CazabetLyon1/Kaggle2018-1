# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
from threading import Thread
import queue
import numpy as np
import math
# ... or use multiprocessing versions
# WARNING: use sentinel based on value, not identity
from multiprocessing import Process


class Generator(object):
    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.steps = math.floor(size/batch_size)


    def __iter__(self):
        pass


class FakeGenerator(Generator):
    def __init__(self, x, y, batch_size = None):
        assert len(x) == len(y)
        size = len(x)
        batch_size = size if batch_size is None else batch_size

        super(FakeGenerator, self).__init__(size, batch_size)

        self.data = [x, y]
        self.batch_size = len(x)

    def __iter__(self):
        if self.steps == 1:
            yield self.data[0], self.data[1]
        else:
            for i in range(self.steps):
                yield self.data[0][i*self.batch_size : (i+1)*self.batch_size], self.data[1][i*self.batch_size : (i+1)*self.batch_size]


class ThreadedGenerator(Generator):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self,
                 size,
                 batch_size,
                 subset_indices,
                 subset_loader,
                 buffer_size=3,
                 x_modifier=None,
                 y_modifier=None):
        super(ThreadedGenerator, self).__init__(size, batch_size)
        self.batch_size = batch_size
        self.subset_loader = subset_loader
        self.subset_indices = subset_indices
        self.buffer_size = buffer_size
        self.exit = False
        self.x_modifier = x_modifier
        self.y_modifier = y_modifier

    def _run(self, buffer):
        buffer_x = []
        buffer_y = []
        for i in self.subset_indices:
            subset_x, subset_y = self.subset_loader(i)
            buffer_x += subset_x
            buffer_y += subset_y
            assert len(buffer_x) == len(buffer_y)

            while len(buffer_x) >= self.batch_size and len(buffer_y) >= self.batch_size and not self.exit:
                batch_x = buffer_x[:self.batch_size]
                if self.x_modifier is not None:
                    batch_x = self.x_modifier(np.array(batch_x))
                batch_y = buffer_y[:self.batch_size]
                if self.y_modifier is not None:
                    batch_y = self.y_modifier(batch_y)

                buffer_x = buffer_x[self.batch_size::]
                buffer_y = buffer_y[self.batch_size::]

                while not self.exit:
                    try:
                        buffer.put((batch_x, batch_y), block=True, timeout=0.10)
                    except queue.Full:
                        pass
                    else:
                        break
            if self.exit:
                break

        if not self.exit:
            buffer.put(None, block=True, timeout=None)

    def __iter__(self):
        print ('gen_iterator_created')
        buffer = queue.Queue(maxsize=self.buffer_size)     # Buffer for the produced data
        thread = Thread(target=self._run, args=(buffer,))  # Producer thread
        thread.start()                                     # Start the producer

        # Loop on all steps
        for i in range(self.steps):
            data = None

            # Read the next produced value
            while not self.exit:
                try:
                    data = buffer.get(block=True, timeout=0.10)
                except queue.Empty:
                    pass
                else:
                    break
            if self.exit or data is None:
                break

            # Return the data for each iteration
            yield data

        # Stop the producing thread
        self.exit = True

        # Empty the buffer
        while True:
            try:
                _ = buffer.get(block=False)
            except queue.Empty:
                break

        # Join the producing thread
        thread.join()

class ThreadedGenerator_bis(Generator):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self,
                 size,
                 batch_size,
                 subset_indices,
                 subset_loader,
                 buffer_size=3,
                 x_modifier=None,
                 y_modifier=None):
        super(ThreadedGenerator, self).__init__(size, batch_size)
        self.batch_size = batch_size
        self.subset_loader = subset_loader
        self.subset_indices = subset_indices
        self.buffer_size = buffer_size
        self.exit = False
        self.x_modifier = x_modifier
        self.y_modifier = y_modifier

    def _run(self, buffer):
        buffer_x = []
        buffer_y = []
        for i in self.subset_indices:
            subset_x, subset_y = self.subset_loader(i)
            buffer_x += subset_x
            buffer_y += subset_y
            assert len(buffer_x) == len(buffer_y)

            while len(buffer_x) >= self.batch_size and len(buffer_y) >= self.batch_size and not self.exit:
                batch_x = buffer_x[:self.batch_size]
                if self.x_modifier is not None:
                    batch_x = self.x_modifier(np.array(batch_x))
                batch_y = buffer_y[:self.batch_size]
                if self.y_modifier is not None:
                    batch_y = self.y_modifier(batch_y)

                buffer_x = buffer_x[self.batch_size::]
                buffer_y = buffer_y[self.batch_size::]

                while not self.exit:
                    try:
                        buffer.put((batch_x, batch_y), block=True, timeout=0.10)
                    except queue.Full:
                        pass
                    else:
                        break
            if self.exit:
                break

        if not self.exit:
            buffer.put(None, block=True, timeout=None)

    def __iter__(self):
        print ('gen_iterator_created')
        buffer = queue.Queue(maxsize=self.buffer_size)     # Buffer for the produced data
        thread = Thread(target=self._run, args=(buffer,))  # Producer thread
        thread.start()                                     # Start the producer

        # Loop on all steps
        for i in range(self.steps):
            data = None

            # Read the next produced value
            while not self.exit:
                try:
                    data = buffer.get(block=True, timeout=0.10)
                except queue.Empty:
                    pass
                else:
                    break
            if self.exit or data is None:
                break

            # Return the data for each iteration
            yield data

        # Stop the producing thread
        self.exit = True

        # Empty the buffer
        while True:
            try:
                _ = buffer.get(block=False)
            except queue.Empty:
                break

        # Join the producing thread
        thread.join()
