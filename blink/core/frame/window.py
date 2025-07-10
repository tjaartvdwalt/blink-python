import math
from itertools import cycle
from queue import Queue

from blink.core.frame.data import FrameData


class FrameWindow(Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize)

    def put(self, item: FrameData, block=True, timeout=None):
        if self.full():
            self.get()

        return super().put(item, block, timeout)

    @property
    def list(self):
        return list(self.queue)

    @property
    def cur(self):
        if self.full():
            l = list(self.queue)
            return l[math.floor(self.maxsize / 2)]

        return None

    @property
    def prev(self):
        if self.full():
            l = list(self.queue)
            return l[math.floor(self.maxsize / 2) - 1]

        return None

    @property
    def first(self):
        if self.full():
            l = list(self.queue)
            return l[0]

        return None

    @property
    def last(self):
        if self.full():
            l = list(self.queue)
            return l[self.maxsize - 1]

        return None
    
    @property
    def size(self):
        return len(self.list)
