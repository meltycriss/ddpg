import argparse

def get_args():
    parser = argparse.ArgumentParser(description='rl')
    # control args
    parser.add_argument('--gpu', type=str, choices=['0', '1', '2', '3'], default='2')
    parser.add_argument('--path', type=str, default='.')

    # model args
    parser.add_argument('--mem_size', type=int)
    parser.add_argument('--lr_critic', type=float)
    parser.add_argument('--lr_actor', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--max_epi', type=int)
    parser.add_argument('--epsilon_decay', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--target_update_frequency', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--random_process', type=bool)
    parser.add_argument('--max_step', type=int)
    args = parser.parse_args()
    return args

def get_model_args():
    args = get_args()
    res = vars(args)
    for key, value in res.items():
        if key=='gpu' or key=='path':
            res.pop(key)
        if value is None:
            res.pop(key)
    return res

def get_control_args():
    args = get_args()
    tmp = vars(args)
    res = {}
    res['gpu'] = tmp['gpu']
    res['path']  =tmp['path']
    return res
