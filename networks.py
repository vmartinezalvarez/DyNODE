import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_FREQ = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True):

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)

class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = QFunction(obs_shape[0], action_shape[0], hidden_dim)
        self.Q2 = QFunction(obs_shape[0], action_shape[0], hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)

def zip_map(zipped, update_op):
    return [update_op(*elems) for elems in zipped]

def euler_update(h_list, dh_list, dt):
    return zip_map(zip(h_list, dh_list), lambda h, dh: h + dt * dh)

def euler_step(func, dt, state, action):
    return euler_update(state, func(state, action), dt)

def rk4_step(func, dt, state, action):
    k1 = func(state, action)
    k2 = func(euler_update(state, k1, dt / 2), action)
    k3 = func(euler_update(state, k2, dt / 2), action)
    k4 = func(euler_update(state, k3, dt), action)

    return zip_map(
        zip(state, k1, k2, k3, k4),
        lambda h, dk1, dk2, dk3, dk4: h + dt * (dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6,)


class DyNODE(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim_p = 200, hidden_dim_r = 200, solver=rk4_step):
        super(DyNODE, self).__init__()
       
        self.trunk_p = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim_p), nn.ELU(),
            nn.Linear(hidden_dim_p, hidden_dim_p), nn.ELU(),
            nn.Linear(hidden_dim_p, hidden_dim_p), nn.ELU(),
            nn.Linear(hidden_dim_p, hidden_dim_p), nn.ELU(),
            nn.Linear(hidden_dim_p, obs_shape[0]))

        self.trunk_r = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, 1), nn.Tanh()) 

        self.apply(weight_init)
        self.dt = torch.tensor(0.001)
        self._solver = solver
  
    def transition(self, obs, actions):

        def _forward_dynamics(_state, _action):
            _obs = _state[0]
            _obs_action = torch.cat([_obs, _action], dim=1)
            return [self.trunk_p(_obs_action)] 

        states = []
        rewards = []
        state = [obs[0]]  

        for obses, action in zip(obs, actions):
            state = self._solver(func=_forward_dynamics, dt=self.dt, state=state, action=action)
            states.append(state[0])
            obs_action = torch.cat([obses, action], dim=1)
            reward = self.trunk_r(obs_action)
            rewards.append(reward)

        return torch.stack(states, dim=0), torch.stack(rewards, dim=0)

    def loss(self, obs_m, action_m, reward_m, next_obs_m):
        obs_seq, reward_hat = self.transition(obs_m, action_m)  
        model_loss = F.mse_loss(obs_seq, next_obs_m)    
        reward_loss = F.mse_loss(reward_m, reward_hat) 
        # reward_loss = torch.mean(torch.abs(reward_m - reward_hat))  
        # model_loss  = torch.mean(torch.abs(obs_seq - next_obs_m)) 
        return model_loss, reward_loss
    
    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        action = action.unsqueeze(0)
        obs = obs.unsqueeze(0)
        next_obs, reward = self.transition(obs, action)
        return reward[-1], next_obs[-1]

class NN_Model(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim_p=200, hidden_dim_r=200, kind='D'):
        super(NN_Model, self).__init__()
        assert kind in ['D', 'P']

        self.trunk_p = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim_p), Swish(),
            nn.Linear(hidden_dim_p, hidden_dim_p), Swish(),
            nn.Linear(hidden_dim_p, hidden_dim_p), Swish(),
            nn.Linear(hidden_dim_p, hidden_dim_p), Swish(),
            nn.Linear(hidden_dim_p, obs_shape[0])) 
        
        if kind == 'P':
            self.trunk_p = nn.Sequential(
                nn.Linear(obs_shape[0] + action_shape[0], hidden_dim_p), nn.ReLU(),
                nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
                nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
                nn.Linear(hidden_dim_p, 2*obs_shape[0])) 

        self.trunk_r = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, hidden_dim_r), nn.ReLU(),
            nn.Linear(hidden_dim_r, 1), nn.Tanh()) 

        self.apply(weight_init)
        self.kind = kind

    def transition(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)

        reward_hat = self.trunk_r(obs_action)

        if self.kind == 'D':
            mean = self.trunk_p(obs_action)
            var = None
        elif self.kind == 'P':
            mean, var = self.trunk_p(obs_action).chunk(2, dim=-1)
            var = F.softplus(var)
        else:
            raise NotImplementedError
        
        return mean, var, reward_hat

    def loss(self, obs, action, reward, next_obs):
        mean, var, reward_hat = self.transition(obs, action)
        reward_loss = F.mse_loss(reward, reward_hat) 

        if self.kind == 'P':
            # Compute the NNL loss
            diff = torch.sub(next_obs, mean)
            model_loss = torch.mean(torch.div(diff**2, var)) + torch.mean(torch.log(var))
        elif self.kind == 'D':
            delta_obs = next_obs - obs            
            model_loss = F.mse_loss(mean, delta_obs)
        else:
            raise NotImplementedError
        return model_loss, reward_loss

    def forward(self, obs, action):
        mean, var, reward_hat = self.transition(obs, action)
        if self.kind == 'P':
            next_obs = mean
        elif self.kind == 'D':
            next_obs = obs + mean
        else:
            raise NotImplementedError
        return reward_hat, next_obs, var
