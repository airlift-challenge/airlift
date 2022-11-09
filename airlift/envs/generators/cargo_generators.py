from typing import List, Collection, NamedTuple, Tuple
import networkx as nx
from gym.utils import seeding
from ordered_set import OrderedSet
from airlift.envs.airport import Airport, AirportID
from airlift.envs.cargo import Cargo, CargoID
from airlift.envs.events.event_generator import EventGenerator
from airlift.envs.route_map import RouteMap
from gym import logger
from airlift.utils.seeds import generate_seed


def generate_cargo_weight():
    """
    Generates a random weight for a cargo. At the moment the weight defaults to 1.
    """
    # return np.random.uniform(low=1.0, high=100.0, size=1)[0]
    return 1


class CargoGenerator:
    """
    Handles the generation of Cargo Tasks
    """

    def __init__(self, num_initial_tasks, max_cargo_per_episode):
        self._np_random = None
        self.routemap = None
        self.num_initial_tasks = num_initial_tasks
        self.max_cargo_per_episode = max_cargo_per_episode
        self.current_cargo_count = 0

    def reset(self, routemap: RouteMap):
        self.routemap = routemap
        self.current_cargo_count = 0

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

    def generate_initial_orders(self) -> set:
        raise NotImplementedError

    def generate_dynamic_orders(self) -> List[Cargo]:
        return []

    def will_generate_more_cargo(self):
        return self.current_cargo_count < self.max_cargo_per_episode


class StaticCargoGenerator(CargoGenerator):
    """
    Handles the generation of static cargo tasks. These are initialized upon the creation of the environment.
    """

    def __init__(self, num_of_tasks=1, soft_deadline_multiplier=1, hard_deadline_multiplier=1.5):
        super().__init__(num_of_tasks, num_of_tasks)

        self._processing_time = None
        self._graph = None
        self.soft_deadline_multiplier = soft_deadline_multiplier
        self.hard_deadline_multiplier = hard_deadline_multiplier
        self.avg_hops = None
        self.avg_flighttime = None

    def reset(self, routemap: RouteMap):
        super().reset(routemap)
        self._processing_time = routemap.airports[0].processing_time  # Assume all processing times are the same

        self.avg_hops = nx.average_shortest_path_length(routemap.multigraph, weight=None) - 1
        self.avg_flighttime = nx.average_shortest_path_length(routemap.multigraph, weight="time")

    def generate_initial_orders(self) -> set:
        """
        Generates static cargo orders upon creation of the environment.

        :Returns:
        -------
        `order_list` : Returns a set that contains all the (static) generated cargo orders
        """

        self.current_cargo_count = self.num_initial_tasks
        return {self.generate_cargo_order(i, self.routemap.drop_off_airports, self.routemap.pick_up_airports) for i in
                range(self.num_initial_tasks)}

    def generate_dynamic_orders(self) -> List[Cargo]:
        return []

    def generate_cargo_order(self, cargo_id, drop_off_airports: OrderedSet[Airport],
                             pickup_airports: OrderedSet[Airport]) -> Cargo:
        """
        Generates cargo orders based on several parameters. Takes into account if we are running a scenario with a
        concentrated drop off or pick up location as well as non-concentrated locations that utilize the entire map.
        The dynamic cargo generator accesses this function without going through the above generate_order function.

        :Parameters:
        ----------
        `cargo_id` : Incremental count of what cargo to create

        `drop_off_airports` : List that contains all the airports that are drop off locations'

        `pick_up_airports` : List that contains all the airports that are pick up locations

        :Returns:
        -------
        `cargo_task` : A fully generated cargo task

        """

        destination_airport = self._np_random.choice(drop_off_airports)
        source_airport = self._np_random.choice(pickup_airports - {destination_airport})

        soft_deadline, hard_deadline = self.create_schedule(source_airport, destination_airport)
        cargo_task = Cargo(cargo_id,
                           source_airport,
                           destination_airport,
                           generate_cargo_weight(),
                           soft_deadline,
                           hard_deadline)
        source_airport.add_cargo(cargo_task)
        return cargo_task

    def create_schedule(self, source, destination) -> Tuple[int, int]:
        """
        Creates a schedule that cargo must be delivered by for it to be considered late (soft deadline) or completely
        missed (hard deadline)

        :Parameters:
        ----------
        `source` : Source Airport

        `destination` : Destination Airport

        :Returns:
        -------
        `soft_deadline, hard_deadline` : Returns a tuple[int, int] containing soft/hard deadlines for cargo.
        """
        # Note for hops - we subtract indices to ignore the starting airport

        shortest_path_hops = nx.shortest_path_length(self.routemap.multigraph, source.id, destination.id,
                                                     weight=None) - 1
        shortest_path_flighttime = nx.shortest_path_length(self.routemap.multigraph, source.id, destination.id,
                                                           weight="time")

        # shortest_path_hops = nx.shortest_path_length(self._graph, source.id, destination.id, weight=None) - 1
        # shortest_path_flighttime = nx.shortest_path_length(self._graph, source.id, destination.id, weight="time")

        # We add one to processing time to account for the time step required to take off
        unit_deadline = ((self._processing_time + 1) * self.avg_hops + self.avg_flighttime) + \
                        ((self._processing_time + 1) * shortest_path_hops + shortest_path_flighttime)

        return round(self.soft_deadline_multiplier * unit_deadline), \
               round(self.hard_deadline_multiplier * unit_deadline)


class DynamicCargoGenerator(StaticCargoGenerator):
    """
    Handles the dynamic generation of cargo using the Poisson process as utilized by Flatlands for train malfunctions
    """

    def __init__(self, cargo_creation_rate: float, max_cargo_to_create: int, num_initial_tasks: int,
                 soft_deadline_multiplier: int = 1,
                 hard_deadline_multiplier: float = 1.5):
        super().__init__(num_of_tasks=num_initial_tasks,
                         soft_deadline_multiplier=soft_deadline_multiplier,
                         hard_deadline_multiplier=hard_deadline_multiplier)

        self.cargo_creation_rate = cargo_creation_rate
        self.max_cargo_per_episode = max_cargo_to_create + num_initial_tasks
        self.eventgen = EventGenerator(self.cargo_creation_rate)

    def generate_dynamic_orders(self) -> List[Cargo]:
        """
        The EventGenerator controls whether a new dynamic cargo is generated or not using poisson distribution. Makes sure that
        the upperbound limit of max cargo generated per episode is also adhered to.

        :Returns:
        ----------
        `cargo_task` : a List that contains a single Cargo or an empty list if no cargo was generated
        """
        if self.eventgen.generate() and \
                self.current_cargo_count < self.max_cargo_per_episode:
            cargo_task = self.generate_cargo_order(self.current_cargo_count,
                                                   self.routemap.drop_off_airports,
                                                   self.routemap.pick_up_airports)
            self.current_cargo_count += 1
            logger.info("Cargo Generated " + "Current " + str(self.current_cargo_count) + " Max: " + str(
                self.max_cargo_per_episode))

            return [cargo_task]

        return []

    def reset(self, routemap):
        super().reset(routemap)

    def seed(self, seed=None):
        super().seed(seed)
        self.eventgen.seed(seed=generate_seed(self._np_random))


class CargoInfo(NamedTuple):
    """Primarily used in testing purposes for the hard coded cargo generator. Properties resemble that of the Cargo class"""
    id: CargoID
    source_airport_id: AirportID
    end_airport_id: AirportID
    weight: float = 1
    soft_deadline: int = 2 ** 32
    hard_deadline: int = 2 ** 32


class HardcodedCargoGenerator(CargoGenerator):
    """
    A basic cargo generator that allows for the initialization of cargo using pre-determined values instead of the
    static or dynamic generators. Generally used for testing purposes.
    """

    def __init__(self, initialcargoinfo=None):
        super().__init__(len(initialcargoinfo), len(initialcargoinfo))
        self._initialcargoinfo: Collection[CargoInfo] = initialcargoinfo
        self.soft_deadline_multiplier = None
        self.hard_deadline_multiplier = None

    def generate_initial_orders(self) -> set:
        """
        Generates the initial hard-coded cargo values.

        :Returns:
        -------
        `initialcargo` : A set that contains the cargo.
        """
        initialcargo = [Cargo(c.id,
                              self.routemap.airports_by_id[c.source_airport_id],
                              self.routemap.airports_by_id[c.end_airport_id],
                              c.weight,
                              c.soft_deadline,
                              c.hard_deadline)
                        for c in self._initialcargoinfo]

        for cargo in initialcargo:
            cargo.source_airport.add_cargo(cargo)

        self.current_cargo_count = len(initialcargo)
        return set(initialcargo)


class NoCargoGenerator(CargoGenerator):
    """
    Initialize an environment with no cargo. Generally used for testing purposes.
    """

    def __init__(self):
        super().__init__(0, 0)
        self.soft_deadline_multiplier = None
        self.hard_deadline_multiplier = None

    def generate_initial_orders(self) -> set:
        return set()