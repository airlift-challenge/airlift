import click
import time
import csv

from airlift.evaluators.utils import doeval_single_episode

# We assume the solution code (i.e., the solution.MySolution class) is under the current directory.
# This is a hack to add the current directory to the Python Path so that MySolution will load.
import sys
sys.path.insert(0, '.')

from solution.mysolution import MySolution

@click.command()
@click.option('--scenario-file',
              default="./scenarios/Test_0/Level_0.pkl",
              help='Location of the evaluation pkl file')
@click.option('--env-seed',
              type=int,
              default=44,
              help='Seed for the environment')
@click.option('--solution-seed',
              type=int,
              default=123,
              help='Seed for the solution')
@click.option('--render/--no-render',
              default=False,
              help='Render the episode')
def run_evaluation(scenario_file, env_seed, solution_seed, render):
    env_info, metrics, time_taken, total_solution_time, metrics = \
        doeval_single_episode(
            scenario_file,
            env_seed,
            MySolution(),
            solution_seed,
            render)

    timestr = time.strftime("%Y-%m-%d-%H%M%S")
    with open("envinfo_{}.csv".format(timestr), 'w', newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(env_info._fields)
        csvwriter.writerow(env_info)
    with open("metrics_{}.csv".format(timestr), 'w', newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(("step",) + metrics[0]._fields)
        for step, metric in enumerate(metrics):
            csvwriter.writerow((step,) + metric)

if __name__ == "__main__":
    run_evaluation()
