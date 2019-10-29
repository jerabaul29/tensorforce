import numpy as np

class NpRingBuffer():
    "A ring buffer using numpy arrays"
    def __init__(self, length, shape, debug=False):
        """Args:
        length: the length of the buffer, int
        shape: the shape of each entry in the buffer, n-dim tuple
        """

        self.length = length
        self.shape = shape

        self.debug = debug

        self.buffer_shape = (length, ) + shape

        self.data = np.zeros(self.buffer_shape, dtype='f')
        self.index_end = -1

        self.n_elements = 0

        if self.debug:
            print(self.index_end)
            print(self.data)

    def push(self, x):
        """Adds array x at the end of the ring buffer. x must be
        of shape shape."""

        assert(x.shape == self.shape), "expect shape {} as input to ring buffer, got {}".format(self.shape, x.shape)

        x_index_end = (self.index_end + 1) % self.length
        self.data[x_index_end] = x
        self.index_end = x_index_end

        self.n_elements = min(self.n_elements + 1, self.length)

        if self.debug:
            print(self.index_end)
            print(self.data)

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index_end - self.n_elements + 1 + np.arange(self.length)) % self.length

        if self.debug:
            print(idx)

        return self.data[idx]
