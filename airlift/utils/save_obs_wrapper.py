import copy
import csv
import os.path
import pickle
from collections import deque
from datetime import datetime
import gym

from airlift.utils.definitions import ROOT_DIR


def check_path(path):
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.memory = deque(maxlen=15000)
        self.file_name_pkl = None
        self.file_name_csv = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        data = [copy.deepcopy(observation), copy.deepcopy(reward), copy.deepcopy(done), copy.deepcopy(info)]
        self.memorize(data)
        return observation, reward, done, info

    def memorize(self, data):
        self.memory.append(data)

    def write_to_pkl(self):
        now = datetime.now()
        pkl_path = ROOT_DIR + '/airlift/saved_obs/pkl/'
        check_path(pkl_path)
        dt_string = now.strftime("Date_%m-%d-%Y_Time_%H-%M-%S")
        self.file_name_pkl = ROOT_DIR + '/airlift/saved_obs/pkl/Obs_Wrapper_' + dt_string + str(id(self)) + '.pkl'
        pickle.dump(self.memory, open(self.file_name_pkl, 'wb'))

    def write_to_csv(self):
        now = datetime.now()
        csv_path = ROOT_DIR + '/airlift/saved_obs/csv/'
        check_path(csv_path)
        dt_string = now.strftime("Date_%m-%d-%Y_Time_%H-%M-%S")
        self.file_name_csv = ROOT_DIR + '/airlift/saved_obs/csv/Obs_Wrapper_' + dt_string + str(id(self)) + '.csv'
        with open(self.file_name_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.memory)
