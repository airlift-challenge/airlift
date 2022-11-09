import numpy as np
from gym.utils import seeding
from airlift.envs.events.event_interval_generator import NoEventIntervalGen
from airlift.envs import ActionHelper, ObservationHelper as oh
from tests.util import generate_environment
from gym import logger

logger.set_level(logger.WARN)


def test_legal_actions(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    num_airports = 4
    env = generate_environment(num_of_airports=num_airports, num_of_agents=1,
                               malfunction_generator=NoEventIntervalGen())
    env.reset(seed)
    obs = env.observe()

    valid_destination(env, num_airports, obs)
    valid_process(env, num_airports, obs)
    load_cargo(env, num_airports)


# unload_cargo(env, num_airports) # Having some issues with this one, i'm probably doing something wrong in the loop.


def valid_process(env, num_airports, obs):
    # Not a fully connected graph, no malfunctions, check to see if we can get to node 4..
    actions = {'a_0': {'process': 5, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': 3}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    env.step(actions)
    assert not action_legal

    # Check the valid process [0,1]
    process = [0, 1]
    for i in range(len(process)):
        actions = {'a_0': {'process': process[i], 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': 3}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        assert action_legal


def valid_destination(env, num_airports, obs):
    # Not a fully connected graph, no malfunctions, check to see if we can get to node 4..
    actions = {'a_0': {'process': 1, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': 3}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    env.step(actions)
    obs = env.observe()
    assert action_legal

    # Check to see if we can get to node > num_airports
    actions = {'a_0': {'process': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': 100}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    obs = env.observe()
    assert not action_legal

    # Go through list of available routes.
    available_routes = obs['a_0']['available_routes']
    while available_routes:
        actions = {'a_0': {'process': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                           'destination': np.random.choice(available_routes)}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        available_routes.pop()
        assert action_legal

    # Go through a list of unavailable routes/out of bounds
    available_routes = [4, 6, 7, 8, 10]
    while available_routes:
        actions = {'a_0': {'process': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                           'destination': np.random.choice(available_routes)}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        available_routes.pop()
        assert not action_legal


def load_cargo(env, num_airports):
    obs = env.observe()
    cargo_info = oh.get_active_cargo_info(env.state(), 0)
    print(cargo_info)

    # Load some random cargo..
    actions = {'a_0': {'process': 0, 'cargo_to_load': [50], 'cargo_to_unload': set(), 'destination': 3}}
    action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
    obs = env.observe()
    assert not action_legal

    while obs['a_0']['current_airport'] is not cargo_info.location:
        obs = env.observe()
        routes_available = obs['a_0']['available_routes']
        destination = np.random.choice(routes_available)
        actions = {'a_0': {'process': 1, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': destination}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        assert action_legal
        env.step(actions)

        if obs['a_0']['current_airport'] == cargo_info.location:
            actions = {'a_0': {'process': 1, 'cargo_to_load': [cargo_info.id], 'cargo_to_unload': set(),
                               'destination': 0}}

            # Load the cargo
            action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
            assert action_legal
            env.step(actions)

            # Attempt to load another cargo at destination thats not there..
            actions = {'a_0': {'process': 1, 'cargo_to_load': [cargo_info.id + 1], 'cargo_to_unload': set(),
                               'destination': destination}}

            action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
            assert not action_legal

    # Go to cargo destination, pick up the cargo


def unload_cargo(env, num_airports):
    obs = env.observe()
    cargo_info = oh.get_active_cargo_info(env.state(), 0)
    while obs['a_0']['current_airport'] is not cargo_info.destination:
        obs = env.observe()
        routes_available = obs['a_0']['available_routes'][0]
        destination = np.random.choice(routes_available)
        actions = {'a_0': {'process': 0, 'cargo_to_load': set(), 'cargo_to_unload': set(), 'destination': destination}}
        action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
        assert action_legal
        env.step(actions)
        if obs['a_0']['current_airport'] == cargo_info.destination:
            actions = {'a_0': {'process': 0, 'cargo_to_load': set(), 'cargo_to_unload': [cargo_info.id],
                               'destination': 1}}

            # Load the cargo
            action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
            assert action_legal
            env.step(actions)

            # Attempt to unload another cargo at destination thats not there..
            actions = {'a_0': {'process': 1, 'cargo_to_load': set(), 'cargo_to_unload': [cargo_info.id + 1],
                               'destination': destination}}

            action_legal, warnings_list = ActionHelper.are_actions_valid(actions, obs)
            assert not action_legal
