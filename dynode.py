import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from networks import Actor, Critic, DyNODE, NN_Model

class DyNODESacAgent(object):
    """DyNODE-SAC."""
    def __init__(self, obs_shape, action_shape, device, model_kind, kind='D', step_MVE=5, hidden_dim=256, discount=0.99, 
        init_temperature=0.01, alpha_lr=1e-3, alpha_beta=0.9, actor_lr=1e-3, actor_beta=0.9, actor_log_std_min=-10, 
        actor_log_std_max=2, critic_lr=1e-3, critic_beta=0.9, critic_tau=0.005,
        critic_target_update_freq=2, model_lr=1e-3, log_interval=100):

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.step_MVE = step_MVE
        self.model_kind = model_kind

        self.actor = Actor(obs_shape, action_shape, hidden_dim, actor_log_std_min, actor_log_std_max).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))

        self.critic = Critic(obs_shape, action_shape, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape) # set target entropy to -|A|
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))
        
        if self.model_kind == 'dynode_model':
            self.model = DyNODE(obs_shape, action_shape, hidden_dim_p=200, hidden_dim_r=200).to(device)  
        elif self.model_kind == 'nn_model':
            self.model = NN_Model(obs_shape, action_shape, hidden_dim_p=200, hidden_dim_r=200, kind=kind).to(device)
        else:
            assert 'model is not supported'

        self.model_optimizer= torch.optim.Adam(self.model.parameters(), lr=model_lr)  

        self.train()
        self.critic_target.train() 

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.model.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_model(self, replay_buffer, L, step):

        if self.model_kind == 'dynode_model':
            obs_m, action_m, reward_m, next_obs_m, _ = replay_buffer.sample_dynode()
            transition_loss, reward_loss = self.model.loss(obs_m, action_m, reward_m, next_obs_m)        
            model_loss = transition_loss + reward_loss
        elif self.model_kind == 'nn_model':
            obs, action, reward, next_obs, _ = replay_buffer.sample()
            transition_loss, reward_loss = self.model.loss(obs, action, reward, next_obs)        
            model_loss = transition_loss + reward_loss        
        else:
            assert 'model is not supported'

        # Optimize the Model
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
    
        if step % self.log_interval == 0:
            L.log('train/model_loss', model_loss, step)
        
    def MVE_prediction(self, replay_buffer, L, step):
        
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        trajectory = []
        next_ob = next_obs
        with torch.no_grad():
            while len(trajectory) < self.step_MVE:
                ob = next_ob
                _, act, _, _ = self.actor(ob)       
                rew, next_ob = self.model(ob, act)
                trajectory.append([ob, act, rew, next_ob])
                
            _, next_action, log_pi, _ = self.actor(next_ob)
            target_Q1, target_Q2 = self.critic_target(next_ob, next_action)
            ret = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi

        critic_loss = 0
        for ob, act, rew, _ in reversed(trajectory):
            current_Q1, current_Q2 = self.critic(ob, act)
            ret = rew + self.discount * ret
            # critic_loss = critic_loss + utils.huber(current_Q1 - ret).mean() + utils.huber(current_Q2 - ret).mean()
            critic_loss = critic_loss + F.mse_loss(current_Q1, ret) + F.mse_loss(current_Q2, ret)
        current_Q1, current_Q2 = self.critic(obs, action)    
        ret = reward + self.discount * ret
        # critic_loss = critic_loss + utils.huber(current_Q1 - ret).mean() + utils.huber(current_Q2 - ret).mean()
        critic_loss = critic_loss + F.mse_loss(current_Q1, ret) + F.mse_loss(current_Q2, ret)
        critic_loss = critic_loss / (self.step_MVE+1)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor 
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs.detach(), pi)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):

        if step < 2000:
            for _ in range(2):
                obs, action, reward, next_obs, not_done = replay_buffer.sample()
                self.update_critic(obs, action, reward, next_obs, not_done, L, step)
                self.update_actor_and_alpha(obs, L, step)
            
            if step % self.log_interval == 0:
                L.log('train/batch_reward', reward.mean(), step)

        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
                        
            if step % self.log_interval == 0:
                L.log('train/batch_reward', reward.mean(), step)

            self.MVE_prediction(replay_buffer, L, step)
            self.update_critic(obs, action, reward, next_obs, not_done, L, step)
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
        
    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step))

    def save_model(self, model_dir, step):
        torch.save(self.model.state_dict(), '%s/model_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(torch.load('%s/critic_%s.pt' % (model_dir, step)))