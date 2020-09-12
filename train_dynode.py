import numpy as np
import torch
import argparse
import os
import gym
gym.logger.set_level(40)
import time
import json
import dmc2gym

import utils
from logger import Logger
from video import VideoRecorder

from dynode import DyNODESacAgent


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)  
    # train
    parser.add_argument('--agent', default='DyNODE-SAC', type=str) 
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=100001, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    # eval
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) 
    parser.add_argument('--critic_target_update_freq', default=1, type=int) 
    # actor                                                                  
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    # model
    parser.add_argument('--model_lr', default=1e-3, type=float)
    parser.add_argument('--kind', default='D', type=str, help= "options D or P") # Deterministic or Probalilistic
    parser.add_argument('--model_num_updates', default=1, type=int)
    parser.add_argument('--model_warm_up', default=1000, type=int)
    parser.add_argument('--k_step', default=10, type=int)   
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='./logdir', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args

def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)
    env = dmc2gym.make(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed) 
    env.seed(args.seed)
    
    method = args.agent + " (H="+ str(args.k_step) +")"

    model_kind = "dynode_model" if args.agent == "DyNODE-SAC" else "nn_model"

    # make directory
    env_name = args.domain_name + '-' + args.task_name
    args.work_dir = args.work_dir + '/' + env_name + '/' + method + '/' + str(args.seed)
    
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape

    replay_buffer = utils.ReplayBuffer(obs_shape=obs_shape, action_shape=action_shape, 
                    capacity=args.replay_buffer_capacity, batch_size=args.batch_size, device=device)

    agent = DyNODESacAgent(obs_shape=obs_shape, action_shape=action_shape, device=device, model_kind = model_kind,
            kind=args.kind, step_MVE = args.k_step, hidden_dim=args.hidden_dim, discount=args.discount,
            init_temperature=args.init_temperature, alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta, actor_lr=args.actor_lr, actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min, actor_log_std_max=args.actor_log_std_max,
            critic_lr=args.critic_lr, critic_beta=args.critic_beta,
            critic_tau=args.critic_tau, critic_target_update_freq=args.critic_target_update_freq,
            model_lr=args.model_lr, log_interval=args.log_interval)
    
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save_model(model_dir, step)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        if step >= args.model_warm_up:
            for _ in range(args.model_num_updates):
                agent.update_model(replay_buffer, L, step)

        # run training update
        if step >= args.init_steps:
            for _ in range(2):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward        
        replay_buffer.add(obs, action, reward, next_obs, done_bool, done)

        obs = next_obs
        episode_step += 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
