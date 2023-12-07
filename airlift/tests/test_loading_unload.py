from airlift.envs import CargoInfo, HardcodedCargoGenerator, NoEventIntervalGen
from airlift.solutions.baselines import ShortestPath
from airlift.tests.util import generate_environment


def test_loading_unloading_order():
    cargo_info = [CargoInfo(id=1, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=2, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=3, source_airport_id=1, end_airport_id=3),
                  CargoInfo(id=4, source_airport_id=1, end_airport_id=3)]

    env = generate_environment(num_of_airports=3,
                               num_of_agents=1,
                               processing_time=5,
                               working_capacity=2,
                               malfunction_generator=NoEventIntervalGen(),
                               cargo_generator=HardcodedCargoGenerator(cargo_info),
                               max_cycles=300)


    _done = False
    solution = ShortestPath()
    obs = env.reset(seed=897)
    solution.reset(obs, seed=4594454)

    while not _done:
        # Compute Action
        actions = solution.policies(env.observe(), env.dones)
        if actions['a_0'] is not None and obs['a_0']['current_airport'] == 1:
            actions['a_0']['cargo_to_load'] = [3,4]
            actions['a_0']['cargo_to_unload'] = [1,2]
        obs, rewards, dones, _ = env.step(actions)  # If there is no observation, just return 0
        print(obs)
        _done = all(dones.values())
