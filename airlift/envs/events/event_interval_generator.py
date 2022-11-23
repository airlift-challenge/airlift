from typing import NamedTuple

from gym.utils import seeding
from airlift.envs.events.event_generator import EventGenerator
from airlift.utils.seeds import generate_seed

EventInterval = NamedTuple('EventInterval', [('num_broken_steps', int)])


class EventIntervalGenerator:
    """
    Generates an interval that a route can go offline. This generator can be used for making anything in the environment
    unavailable from min to max steps inclusive. At the moment it is only being utilized to manage routes becoming unavailable.
    This generator is based upon the Flatland Train Malfunction generator and uses poisson distribution as defined in the
    EventGenerator class.
    """
    def __init__(self, malfunction_rate: float, min_duration: int, max_duration: int):
        self.eventgen = EventGenerator(malfunction_rate)
        self.malfunction_rate = malfunction_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._np_random = None
        self._randcache = []

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.eventgen.seed(seed=generate_seed(self._np_random))

    def generate(self) -> EventInterval:
        """
        Generates an Event with an Interval from min duration to max duration

        :return: `EventInterval`: a NamedTuple that contains the number of steps something will become unavailable for. The number of
            steps is 0 of an interval wasn't generated.

        """
        if self.eventgen.generate():
            if not self._randcache:
                self._randcache = list(self._np_random.integers(self.min_duration, self.max_duration + 1, size=1000))
            num_broken_steps = self._randcache.pop() + 1
        else:
            num_broken_steps = 0
        return EventInterval(num_broken_steps)


class NoEventIntervalGen(EventIntervalGenerator):
    """
    Used when items that utilize the EventIntervalGenerator are toggled off.
    """
    def __init__(self):
        super().__init__(0, 0, 0)