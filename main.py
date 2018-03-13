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
from ddpg import arguments

# suppress INFO level logging 'Starting new video recorder writing to ...'
logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.WARNING)
# suppress INFO level logging 'Creating monitor directory ...'
logging.getLogger('gym.wrappers.monitoring').setLevel(logging.WARNING)

ENV_NAME = 'foa-v0'
env = gym.make(ENV_NAME)

control_args = arguments.get_control_args()

os.environ["CUDA_VISIBLE_DEVICES"] = control_args['gpu']

root = control_args['path']
if root is None:
    root = ENV_NAME
if not os.path.exists(root):
    os.mkdir(root)

model_args = arguments.get_model_args()

args = [
        model_args,
        ]
# model_names is a list like [max_epi_10, max_epi_20]
model_names = [
        '_'.join(
        [
        '_'.join(
        [key, str(value)])
        for key, value in arg.items()])
        for arg in args]
# handle standard arg, i.e., {}
model_names = ['standard' if name=='' else name  for name in model_names]
    

# model loop
for i in trange(len(args), desc='model', leave=True):
    model_dir = '{}/{}'.format(root, model_names[i])
    os.mkdir(model_dir)
    arg = args[i]
    # repeat loop
    for n in trange(control_args['repeat'], desc='repeat', leave=True):
        dir = '{}/{}'.format(model_dir, n)
        ddpg=DDPG(env, **arg)
        ddpg.train(dir)
        ddpg.save(dir)
        ddpg.test(dir, n=control_args['n_test'])





