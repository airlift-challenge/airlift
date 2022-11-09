import math

import click

from airlift.envs import PlaneType
from airlift.envs.events.event_interval_generator import EventIntervalGenerator, NoEventIntervalGen
from airlift.envs.generators.cargo_generators import StaticCargoGenerator, DynamicCargoGenerator
from airlift.utils.definitions import ROOT_DIR
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator, LimitedDropoffEntryRouteGenerator
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator, PlainMapGenerator
from airlift.evaluators.utils import ScenarioInfo, generate_scenarios
from pathlib import Path






@click.command()
@click.option('--output_path',
              type=click.Path(file_okay=False, exists=False, path_type=Path),
              default=Path(ROOT_DIR + "/scenarios/"),
              help='Where to put the scenario pkl files')
@click.option('--run-random/--no-run-random', is_flag=True, show_default=True, default=True,
                help="Run random algorithm to get score")
@click.option('--run-baseline/--no-run-baseline', is_flag=True, show_default=True, default=True,
                help="Run baseline algorithm to get score")
def create_scenarios_based_on_test_level(output_path=Path(ROOT_DIR + "/scenarios/"),
                                         run_random=True,
                                         run_baseline=True):
    multiprocess = True

    # Initialize start values...
    # Scenarios generated_debug (text*levels)
    number_of_tests = 8
    number_of_levels = 4

    # Base Values
    num_airports = 10
    num_agents = 1
    processing_time = 10

    num_cargo = 1
    soft_deadline_multiplier = 25
    hard_deadline_multiplier = 50
    num_airports_increase = 4
    starting_cargo_per_airport = 6
    starting_airplanes_per_airport = 2


    # List of ScenarioInfo
    scenarios = []

    #
    single_plane_type = [PlaneType(id=0, max_range=2, speed=0.4, max_weight=10)]

    multiple_plane_types = [PlaneType(id=0, max_range=2, speed=0.2, max_weight=20),
                            PlaneType(id=1, max_range=1, speed=0.5, max_weight=3)]

    # For each test

    for test in range(number_of_tests): # One plane type, no zones
        testset = test // 2
        testnum = test % 2

        # Calculate new values for next level
        max_working_capacity = 10
        working_capacity = 2
        num_airports = 10 ** (testnum+1)
        num_cargo = math.ceil(starting_cargo_per_airport * num_airports)
        num_agents = math.ceil(starting_airplanes_per_airport * num_airports)
        num_drop_off_airports = math.ceil(math.log(num_airports))
        num_pick_up_airports = num_drop_off_airports

        for level in range(number_of_levels):
            levelset = level // 2
            num_dynamic_cargo = levelset * 5
            dynamic_cargo_soft_deadline_multiplier = 5
            dynamic_cargo_hard_deadline_multiplier = 15
            dynamic_cargo_generation_rate = 1/100

            malfunction_rate = levelset * 1 / 300
            min_duration = 10
            max_duration = 100

            if testset == 0:
                scenarios.append(ScenarioInfo(test,
                                              level,
                                              AirliftEnv(
                                                  world_generator=AirliftWorldGenerator(
                                                      plane_types=single_plane_type,
                                                      airport_generator=RandomAirportGenerator(
                                                          max_airports=num_airports,
                                                          processing_time=processing_time,
                                                          working_capacity=working_capacity,
                                                          make_drop_off_area=False,
                                                          make_pick_up_area=False,
                                                          num_drop_off_airports=1,
                                                          num_pick_up_airports=1
                                                      ),
                                                      route_generator=RouteByDistanceGenerator(
                                                          malfunction_generator=EventIntervalGenerator(
                                                              malfunction_rate=malfunction_rate,
                                                              min_duration=min_duration,
                                                              max_duration=max_duration),
                                                          route_ratio=2
                                                      ),
                                                      cargo_generator=DynamicCargoGenerator(
                                                          cargo_creation_rate=dynamic_cargo_generation_rate,
                                                          max_cargo_to_create=num_dynamic_cargo,
                                                          num_initial_tasks=num_cargo,
                                                          soft_deadline_multiplier=soft_deadline_multiplier,
                                                          hard_deadline_multiplier=hard_deadline_multiplier
                                                      ),
                                                      airplane_generator=AirplaneGenerator(num_agents),
                                                      max_cycles=5000
                                                  )
                                              )
                                              )
                                 )
            elif testset == 1:
                scenarios.append(ScenarioInfo(test,
                                              level,
                                              AirliftEnv(
                                                  world_generator=AirliftWorldGenerator(
                                                      plane_types=single_plane_type,
                                                      airport_generator=RandomAirportGenerator(
                                                          max_airports=num_airports,
                                                          processing_time=processing_time,
                                                          working_capacity=working_capacity,
                                                          make_drop_off_area=True,
                                                          make_pick_up_area=True,
                                                          num_drop_off_airports=num_drop_off_airports,
                                                          num_pick_up_airports=num_pick_up_airports
                                                      ),
                                                      route_generator=RouteByDistanceGenerator(
                                                          malfunction_generator=EventIntervalGenerator(
                                                              malfunction_rate=malfunction_rate,
                                                              min_duration=min_duration,
                                                              max_duration=max_duration),
                                                          route_ratio=2
                                                      ),
                                                      cargo_generator=DynamicCargoGenerator(
                                                          cargo_creation_rate=dynamic_cargo_generation_rate,
                                                          max_cargo_to_create=num_dynamic_cargo,
                                                          num_initial_tasks=num_cargo,
                                                          soft_deadline_multiplier=soft_deadline_multiplier,
                                                          hard_deadline_multiplier=hard_deadline_multiplier
                                                      ),
                                                      airplane_generator=AirplaneGenerator(num_agents),
                                                      max_cycles=5000
                                                  )
                                              )
                                              )
                                 )
            elif testset == 2:
                scenarios.append(ScenarioInfo(test,
                                              level,
                                              AirliftEnv(
                                                  world_generator=AirliftWorldGenerator(
                                                      plane_types=multiple_plane_types,
                                                      airport_generator=RandomAirportGenerator(
                                                          max_airports=num_airports,
                                                          processing_time=processing_time,
                                                          working_capacity=working_capacity,
                                                          make_drop_off_area=True,
                                                          make_pick_up_area=True,
                                                          num_drop_off_airports=num_drop_off_airports,
                                                          num_pick_up_airports=num_pick_up_airports
                                                      ),
                                                      route_generator=RouteByDistanceGenerator(
                                                          malfunction_generator=EventIntervalGenerator(
                                                              malfunction_rate=malfunction_rate,
                                                              min_duration=min_duration,
                                                              max_duration=max_duration),
                                                          route_ratio=2
                                                      ),
                                                      cargo_generator=DynamicCargoGenerator(
                                                          cargo_creation_rate=dynamic_cargo_generation_rate,
                                                          max_cargo_to_create=num_dynamic_cargo,
                                                          num_initial_tasks=num_cargo,
                                                          soft_deadline_multiplier=soft_deadline_multiplier,
                                                          hard_deadline_multiplier=hard_deadline_multiplier
                                                      ),
                                                      airplane_generator=AirplaneGenerator(num_agents),
                                                      max_cycles=5000
                                                  )
                                              )
                                              )
                                 )
            elif testset == 3:
                scenarios.append(ScenarioInfo(test,
                                              level,
                                              AirliftEnv(
                                                  world_generator=AirliftWorldGenerator(
                                                      plane_types=multiple_plane_types,
                                                      airport_generator=RandomAirportGenerator(
                                                          max_airports=num_airports,
                                                          processing_time=processing_time,
                                                          working_capacity=working_capacity,
                                                          make_drop_off_area=True,
                                                          make_pick_up_area=True,
                                                          num_drop_off_airports=num_drop_off_airports,
                                                          num_pick_up_airports=num_pick_up_airports
                                                      ),
                                                      route_generator=LimitedDropoffEntryRouteGenerator(
                                                          malfunction_generator=EventIntervalGenerator(
                                                              malfunction_rate=malfunction_rate,
                                                              min_duration=min_duration,
                                                              max_duration=max_duration),
                                                          route_ratio=2,
                                                          drop_off_fraction_reachable=0.2,
                                                          pick_up_fraction_reachable=0.2),
                                                      cargo_generator=DynamicCargoGenerator(
                                                          cargo_creation_rate=dynamic_cargo_generation_rate,
                                                          max_cargo_to_create=num_dynamic_cargo,
                                                          num_initial_tasks=num_cargo,
                                                          soft_deadline_multiplier=soft_deadline_multiplier,
                                                          hard_deadline_multiplier=hard_deadline_multiplier
                                                      ),
                                                      airplane_generator=AirplaneGenerator(num_agents),
                                                      max_cycles=5000
                                                  )
                                              )
                                              )
                                 )
            else:
                assert False, "No more tests to generate"

    generate_scenarios(output_path,
                       scenarios,
                       multiprocess=multiprocess,
                       run_random=run_random,
                       run_baseline=run_baseline)


if __name__ == "__main__":
    create_scenarios_based_on_test_level()
