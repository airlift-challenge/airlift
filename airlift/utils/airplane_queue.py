"""
This is an implementation of the PriorityQueue https://github.com/python/cpython/blob/3.11/Lib/queue.py for the Airlift Environment
"""
import heapq
import types
from collections import deque
from heapq import heappush, heappop
from time import monotonic as time

try:
    from _queue import SimpleQueue
except ImportError:
    SimpleQueue = None

__all__ = ['Empty', 'Full', 'Queue', 'AirplaneQueue']

try:
    from _queue import Empty
except ImportError:
    class Empty(Exception):
        'Exception raised by Queue.get(block=0)/get_nowait().'
        pass


class Full(Exception):
    'Exception raised by Queue.put(block=0)/put_nowait().'
    pass


class Queue:
    """
    Create a queue object with a given maximum size.
    If maxsize is <= 0, the queue size is infinite.
    """

    def __init__(self, maxsize=0):
        self.unfinished_tasks = 0
        self.maxsize = maxsize
        self._init(maxsize)

    def qsize(self):
        """
        Return the approximate size of the queue (not reliable!).
        """
        return self._qsize()

    def empty(self):
        """
        Return True if the queue is empty, False otherwise (not reliable!).
        """
        return not self._qsize()

    def full(self):
        """
        Return True if the queue is full, False otherwise (not reliable!).
        """
        return 0 < self.maxsize <= len(self.queue)

    def task_done(self):
        unfinished = self.unfinished_tasks - 1
        if unfinished <= 0:
            if unfinished < 0:
                raise ValueError('task_done() called too many times')
        self.unfinished_tasks = unfinished

    def put(self, item, block=True, timeout=None):
        """
            Put an item into the queue.
        """
        if self.maxsize > 0:
            if not block:
                if len(self.queue) >= self.maxsize:
                    raise Full
            elif timeout is None:
                while len(self.queue) >= self.maxsize:
                    pass
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while len(self.queue) >= self.maxsize:
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Full
                    pass

        self._put(item)
        self.unfinished_tasks += 1

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.
        """
        if not block:
            if not self.queue:
                raise Empty
        elif timeout is None:
            while not self.queue:
                pass
        elif timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        else:
            endtime = time() + timeout
            while not self.queue:
                remaining = endtime - time()
                if remaining <= 0.0:
                    raise Empty
                pass
        self.task_done()
        return self._get()

    def put_nowait(self, item):
        """Put an item into the queue without blocking.
        """
        return self.put(item, block=False)

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.
        """
        return self.get(block=False)

    def _init(self, maxsize):
        self.queue = deque()

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.popleft()

    __class_getitem__ = classmethod(types.GenericAlias)


class AirplaneQueue(Queue):
    """
    Variant of Queue that retrieves open entries in priority order (lowest first).
    Entries are typically tuples of the form:  (priority number, data).
    """

    def __init__(self):
        super().__init__()
        self.added_agents = set()

    def peek_at_next(self):
        return self.queue[0] if self.queue else None

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        heappush(self.queue, item)

    def _get(self):
        return heappop(self.queue)

    def put_nowait(self, item):
        """Put an item into the queue without blocking.
        """
        return self.put(item, block=False)

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.
        """
        return self.get(block=False)

    # Lets avoid iterating comparison of (self.priority, agent) in queue and instead keep track of it using a set()?
    def add_to_waiting_queue(self, agent, count):
        if agent not in self.added_agents:
            self.put((agent.priority, count, agent))
            self.added_agents.add(agent)

    # Once agents are done waiting/processing, they are removed through the airport class
    def agent_complete(self, agent):
        if agent in self.added_agents:
            self.added_agents.remove(agent)

    def is_added_agents_empty(self):
        if not self.added_agents:
            return True
        else:
            return False

    def update_priority(self, old_priority, new_priority, old_count, airport_counter, agent):

        entry_to_update = (old_priority, old_count, agent)
        if entry_to_update in self.queue:
            agent.priority = new_priority
            new_entry = (new_priority, next(airport_counter), agent)
            # Add the new entry
            self.put(new_entry)

            # Remove the old entry
            self.queue.remove(entry_to_update)
            current_size = self._qsize()
            self.unfinished_tasks = current_size
            heapq.heapify(self.queue)

    def is_agent_in_queue(self, agent):
        for item in self.queue:
            if item[2] == agent:
                return True, item
            else:
                return False, None
