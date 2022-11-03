from enum import IntEnum
from typing import Tuple, List, Set, Dict, Collection
from gym import logger

from airlift.envs.plane_types import PlaneType
from airlift.envs.cargo import Cargo, CargoID
from airlift.envs.world_map import Coordinate
from airlift.envs.route_map import RouteMap
from airlift.envs.airport import NOAIRPORT_ID, Airport

AgentID = str


class PlaneState(IntEnum):
    """
    Enumeration that defines the states an agent can be in

        - :0: Waiting
        - :1: Processing
        - :2: Stopped
        - :3: Moving
        - :4: Ready for takeoff
    """

    # Docstring may seem redundant here, but its mainly for the apidocs
    WAITING = 0
    PROCESSING = 1
    STOPPED = 2
    MOVING = 3
    READY_FOR_TAKEOFF = 4


class EnvAgent:
    """
    Represents an individual agent (airplane) in an environment.
    """

    def __init__(self,
                 start_airport,
                 routemap=None,
                 plane_type=None,
                 max_loaded_weight=None,
                 malfunction_generator=None):

        self.flight_cost = 0
        self.flight_start_position: Coordinate = None
        self.destination_airport: Airport = None
        self.current_airport: Airport = start_airport
        self.routemap: RouteMap = routemap
        self.total_flight_time = None

        self.state = PlaneState.READY_FOR_TAKEOFF

        self.cargo_being_loaded: Set[Cargo] = set()
        self.cargo_being_unloaded: Set[Cargo] = set()

        self.elapsed_flight_time = None

        self.processing_time_left = 0
        self.cargo: Set[Cargo] = set()
        self.waiting_steps = 0

        self.last_direction = None

        self.max_loaded_weight = int(max_loaded_weight)
        self.plane_type: PlaneType = plane_type

        self.previous_airport = None

        self.malfunction_generator = malfunction_generator

    def _handle_waiting(self, action: dict, cargo_by_id: Dict[CargoID, Cargo], elapsed_time, warnings: List[str]) -> None:
        """
        Handles trnasitions from the waiting state

        Parameters
        ----------
        action - A dictionary containing 'process', 'cargo to load' cargo to unload' and destination
        cargo_by_id - Dictionary containing the Cargo and CargoID
        warnings - List of warnings issued by the environment
        """
        if action["process"]:
            success = self.try_to_process([cargo_by_id[id] for id in action["cargo_to_load"]],
                                          [cargo_by_id[id] for id in action["cargo_to_unload"]],
                                          elapsed_time,
                                          warnings)
            if success:
                action["process"] = 0
                action["cargo_to_load"] = []
                action["cargo_to_unload"] = []

    def _handle_processing(self) -> None:
        """
        Handles trnasitions from the processing state
        """
        if self.processing_time_left > 0:
            self.processing_time_left -= 1
        else:
            self.cargo = self.cargo.union(self.cargo_being_loaded)
            self.cargo_being_loaded = set()

            for c in self.cargo_being_unloaded:
                self.current_airport.add_cargo(c)
            self.cargo_being_unloaded = set()

            self.current_airport.remove_from_capacity(self)
            self.state = PlaneState.READY_FOR_TAKEOFF

    def _handle_moving(self) -> None:
        """
        Handles transitions from the moving state.
        """
        self.elapsed_flight_time += 1
        if self.elapsed_flight_time > self.total_flight_time:
            self.current_airport = self.destination_airport
            self.destination_airport = None
            self.state = PlaneState.WAITING
            self.elapsed_flight_time = None

    def _handle_ready_for_takeoff(self, action, cargo_by_id: Dict[CargoID, Cargo], elapsed_time, warnings: List[str]) -> None:
        """
        Handles transition when when ready for take off. Checks if routes are reachable from the current airport,
        checks for routes being made unavailable due to malfunctions, gets the flight time and flight cost associated
        with the edge being taken. After the agent starts moving the destination is set to the default value of NOAIRPORT_ID
        which is set to 0.

        :Parameters:
        ----------
        `action` : A dictionary of actions that contains 'process', 'cargo_to_load', 'cargo_to_unload' and destination
        `cargo_by_id` : Dictionary containing the Cargo and CargoID
        `warnings` : List of warnings issued by the environment. Ex: If an action is given to an unavailable route
        """
        if action["process"]:
            success = self.try_to_process([cargo_by_id[id] for id in action["cargo_to_load"]],
                                          [cargo_by_id[id] for id in action["cargo_to_unload"]],
                                          elapsed_time,
                                          warnings)
            if success:
                action["process"] = 0
                action["cargo_to_load"] = []
                action["cargo_to_unload"] = []
        elif action["destination"] != NOAIRPORT_ID:
            new_destination = self.routemap.airports_by_id[action["destination"]]

            if not self.routemap.reachable(self.current_airport, self.routemap.airports_by_id[action["destination"]],
                                           self.plane_type):
                warnings.append("The destination airport is not reachable from here!")
                action["destination"] = NOAIRPORT_ID
            elif action["destination"] not in self.routemap.get_available_routes(self.current_airport,
                                                                                 self.plane_type):
                warnings.append("ROUTE FROM: " + str(self.current_airport.id) + " TO: " + str(new_destination.id) + \
                                " is currently unavailable. Change destination or wait for it to become available! The route will be unavailable for "
                                + str(
                    self.routemap.get_malfunction_time(self.current_airport.id, new_destination.id,
                                                       self.plane_type)) + " steps!")
            else:
                self.elapsed_flight_time = 0

                self.destination_airport = new_destination

                flight_time = self.routemap.get_flight_time(self.current_airport,
                                                            new_destination, self.plane_type)

                self.flight_cost += self.routemap.get_flight_cost(self.current_airport,
                                                                  new_destination, self.plane_type)

                self.flight_start_position = self.current_airport.position

                self.state = PlaneState.MOVING

                self.total_flight_time = flight_time
                self.previous_airport = self.current_airport

                action["destination"] = NOAIRPORT_ID

    @staticmethod
    def _check_cargo(cargo_ids: Collection[CargoID], cargo_by_id: Dict[CargoID, Cargo], warnings: List[str]):
        """
        Filters out invalid cargo from the given cargo collection.

        :Parameters:
        ----------
        `cargo_ids` : A list of CargoIDs

        `cargo_by_id` : Dictionary containing all the cargo and IDs

        `warnings` : A list containing any warnings. In this case if a particular cargo ID does not exist.

        """
        valid_ids = []
        for id in cargo_ids:
            if id not in cargo_by_id:
                warnings.append("Cargo {} does not exist".format(id))
            else:
                valid_ids.append(id)

        return valid_ids

    def step(self, action, cargo_by_id: Dict[CargoID, Cargo], elapsed_time) -> Tuple[dict, List[str]]:
        """
        Updates the agent's state.

        :Parameters:
        ----------
        `action` : A dictionary of actions that contains 'process', 'cargo_to_load', 'cargo_to_unload' and 'destination'

        `cargo_by_id` : Dictionary containing the Cargo and CargoID

        :Returns:
        -------
        `updated_action` : The updated action contains any updated events to the action dict that occurred during the time
        step

        `warnings` : A list containing any issues with the current actions given.
        """
        self.last_state = self.state
        warnings: List[str] = []
        updated_action = action.copy()
        updated_action["cargo_to_load"] = self._check_cargo(updated_action["cargo_to_load"], cargo_by_id, warnings)
        updated_action["cargo_to_unload"] = self._check_cargo(updated_action["cargo_to_unload"], cargo_by_id, warnings)

        if self.state == PlaneState.READY_FOR_TAKEOFF:
            self._handle_ready_for_takeoff(updated_action, cargo_by_id, elapsed_time, warnings)
        elif self.state == PlaneState.MOVING:
            self._handle_moving()
        elif self.state == PlaneState.PROCESSING:
            self._handle_processing()
        elif self.state == PlaneState.WAITING:
            self.waiting_steps += 1
            self._handle_waiting(updated_action, cargo_by_id, elapsed_time, warnings)

        return updated_action, warnings

    def try_to_process(self,
                       cargo_to_load: Collection[Cargo],
                       cargo_to_unload: Collection[Cargo],
                       elapsed_time,
                       warnings: List[str]) -> bool:
        """
        Checks to see if the current airport has capacity. If there is capacity the agent will go into
        the PROCESSING state and the processing timer will be updated. The agent is added to the airports capacity.

        :Parameters:
        ----------
        `cargo_to_load` : A set that contains cargo to load

        `cargo_to_unload` : A set that contains cargo to unload

        :Returns:
        -------
        `success` : If successfully load or unloaded cargo

        """
        success = False
        if self.current_airport.airport_has_capacity():
            self.load_cargo(cargo_to_load, elapsed_time, warnings)
            self.unload_cargo(cargo_to_unload, warnings)

            self.processing_time_left = self.current_airport.processing_time
            self.current_airport.add_to_capacity(self)
            self.state = PlaneState.PROCESSING
            success = True

        return success

    def load_cargo(self, cargo_to_load: Collection[Cargo], elapsed_steps, warnings: List[str]):
        """
        Checks to make sure airplane can load cargo and loads the cargo. Also checks to ensure that cargo is assigned to
        that airplane

        :Parameters:
        ----------
        `cargo_to_load` : A list that contains the Cargo to load

        `warnings` : List of warnings issued by the environment. Ex: If an action is given to an unavailable route
        """

        for cargo in cargo_to_load:
            if cargo not in self.current_airport.cargo:
                warnings.append(
                    "Unable to load Cargo ID: " + str(cargo.id) + " is not at airport")
            elif cargo.earliest_pickup_time > elapsed_steps:
                time_rem = cargo.earliest_pickup_time - elapsed_steps
                print(
                    "Unable to load Cargo ID: " + str(cargo.id) +
                    ". The cargo is not ready yet and will be ready in " + str(time_rem) + " steps!")
            else:
                if (cargo.weight + self.current_cargo_weight) <= self.max_loaded_weight:
                    self.current_airport.remove_cargo(cargo)
                    self.cargo_being_loaded.add(cargo)
                else:
                    warnings.append(
                        "Unable to load Cargo ID: " + str(cargo.id) + " due to exceeding allowed airplane weight limit")

    def unload_cargo(self, cargo_to_unload: Collection[Cargo], warnings: List[str]):
        """
        Does a check to ensure the correct cargo is being unloaded and removes the cargo from the airplane.

        :Parameters:
        ----------
        `cargo_to_unload` : A list that contains the Cargo to unload

        `warnings` : List of warnings issued by the environment. Ex: If an action is given to an unavailable route
        """

        for cargo in cargo_to_unload:
            if cargo in self.cargo:
                self.cargo.remove(cargo)
                self.cargo_being_unloaded.add(cargo)
                logger.debug("Airplane Type: " + str(self.plane_type.id) + " delivered cargo!")
            else:
                warnings.append("Unable to unload cargo, ID: {}, AIRPORT: {}"
                                .format(cargo.id, self.current_airport.id))

    @property
    def current_cargo_weight(self) -> float:
        """
        Gets the total current cargo weight on an airplane

        :Returns:
        -------
        `weight` : current cargo weight on an airplane
        """
        weight = sum(c.weight for c in self.cargo)
        return weight
