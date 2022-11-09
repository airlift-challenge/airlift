import pytest

from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.airlift_env import ActionHelper
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.evaluators.local_evaluator import LocalEvaluationService
from airlift.evaluators.utils import doeval
from airlift.solutions.baselines import RandomAgent
from tests.util import generate_environment
from airlift.evaluators.utils import ScenarioInfo, generate_scenarios

# Generates the test folder with a new environment pickle file
@pytest.fixture()
def test_env_folder(tmp_path):
    test_root = tmp_path / "test-folder"

    scenarios = []
    scenarios.append(ScenarioInfo(0, 0, generate_environment(num_of_agents=1, cargo_generator=StaticCargoGenerator(1), num_of_airports=6, processing_time=1, malfunction_generator=EventIntervalGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=2))))
    scenarios.append(ScenarioInfo(0, 1, generate_environment(num_of_agents=2, cargo_generator=StaticCargoGenerator(2), num_of_airports=6, processing_time=1, malfunction_generator=EventIntervalGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=2))))
    scenarios.append(ScenarioInfo(1, 0, generate_environment(num_of_agents=3, cargo_generator=StaticCargoGenerator(3), num_of_airports=10, processing_time=1, malfunction_generator=EventIntervalGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=2))))
    scenarios.append(ScenarioInfo(1, 1, generate_environment(num_of_agents=4, cargo_generator=StaticCargoGenerator(4), num_of_airports=10, processing_time=1, malfunction_generator=EventIntervalGenerator(malfunction_rate=1 / 5, min_duration=1, max_duration=2))))

    generate_scenarios(test_root, scenarios)
    return test_root


def test_local_service(test_env_folder):
    doeval(test_env_folder, RandomAgent())
