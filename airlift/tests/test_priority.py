import random

import numpy as np
from airlift.envs.agents import EnvAgent
from gym.utils import seeding
from airlift.envs.events.event_interval_generator import NoEventIntervalGen
from airlift.envs import ActionHelper, ObservationHelper as oh, HardcodedCargoGenerator, NOAIRPORT_ID, CargoInfo, \
    StaticCargoGenerator
from airlift.tests.util import generate_environment
from gym import logger

from airlift.utils.airplane_queue import AirplaneQueue

logger.set_level(logger.WARN)


def test_priority_actions(render):
    seed = 546435
    np_random, seed = seeding.np_random(seed)
    num_airports = 3
    cargo_info = [CargoInfo(id=0, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=1, source_airport_id=3, end_airport_id=1),
                  CargoInfo(id=2, source_airport_id=3, end_airport_id=1)]
    env = generate_environment(num_of_airports=num_airports, num_of_agents=3, processing_time=0,
                               malfunction_generator=NoEventIntervalGen(),
                               cargo_generator=HardcodedCargoGenerator(cargo_info))
    obs = env.reset(seed)
    agent_actions(env, obs)


def test_agent_priority():
    seed = 42
    random.seed(seed)

    pq = AirplaneQueue()
    # Add an initial queue to the pq
    for i in range(1000):
        pq.put(EnvAgent(start_airport=1, max_loaded_weight=10, priority=random.randint(1, 100)))

    added_agents = 0
    while not pq.empty():

        agent = pq.get()
        next_agent = pq.peek_at_next()
        if not pq.empty():
            assert next_agent >= agent

        # Add agents to the queue during execution
        if added_agents <= 100:
            added_agents += 1
            prob = random.uniform(0, 1)
            if prob > .5:
                pq.put(EnvAgent(start_airport=1, max_loaded_weight=10, priority=random.randint(1, 100)))


def agent_actions(env, obs):
    done = False
    cargo = 0
    prio = 1

    while not done:
        available_routes = obs['a_0']['available_routes']
        route = np.random.choice(available_routes)
        actions = {a: None for a in env.agents}

        for key in actions.keys():
            if obs['a_0']['cargo_at_current_airport']:
                actions[key] = {'priority': prio, 'cargo_to_load': [cargo], 'cargo_to_unload': set(),
                                'destination': route}
                cargo += 1
                if prio <= 3:
                    prio += 1
            elif obs['a_0']['current_airport'] == 1:
                actions[key] = {'priority': None, 'cargo_to_load': set(), 'cargo_to_unload': [cargo],
                                'destination': route}
            else:
                actions[key] = {'priority': None, 'cargo_to_load': set(), 'cargo_to_unload': set(),
                                'destination': 1}

        # print(obs)
        obs, rewards, dones, _ = env.step(actions)
        done = all(dones.values())
