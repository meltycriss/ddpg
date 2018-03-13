import gym
import gym_foa
from ddpg import DDPG
from tqdm import trange
import os
import shutil
from ddpg import util
from ddpg import common
import sys
import logging

assert len(sys.argv)>1, 'please specify path!'

root = sys.argv[1]

# model level
# train data
df = util.concat_models(root, csv_name='train_data.csv')
util.plot(df[df[common.S_EPI] > 0], dir=root, name='train_data.png')
util.plot(df[df[common.S_EPI] % 10 == 0], dir=root, name='train_data_gap_10.png')
util.plot(df[df[common.S_EPI] % 100 == 0], dir=root, name='train_data_gap_100.png')
# test data
df = util.concat_models(root, csv_name='test_data.csv')
util.plot(df[df[common.S_EPI] > 0], dir=root, name='test_data.png')

# repeat level
subfolders = next(os.walk(root))[1]
for subfolder in subfolders:
    path = os.path.join(root, subfolder)
    df = util.concat_times(path, csv_name='train_data.csv')
    df[common.S_MODEL] = subfolder
    util.plot(df[df[common.S_EPI] > 0], dir=path, name='train_data.png')
    util.plot(df[df[common.S_EPI] % 10 == 0], dir=path, name='train_data_gap_10.png')
    util.plot(df[df[common.S_EPI] % 100 == 0], dir=path, name='train_data_gap_100.png')
    df = util.concat_times(path, csv_name='test_data.csv')
    df[common.S_MODEL] = subfolder
    util.plot(df[df[common.S_EPI] > 0], dir=path, name='test_data.png')




