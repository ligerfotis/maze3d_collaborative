import torch
import numpy as np
from rl_models.networks_discrete import update_params, Actor, Critic, ReplayBuffer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiscreteSACAgent:
    def __init__(self, config=None, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, buffer_max_size=1000000, tau=0.005,
                 update_interval=1, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 chkpt_dir=None, target_entropy_ratio=0.4):
        if config is not None:
            # SAC params
            self.batch_size = config['SAC']['batch_size']
            self.layer1_size = config['SAC']['layer1_size']
            self.layer2_size = config['SAC']['layer2_size']
            self.gamma = config['SAC']['gamma']
            self.tau = config['SAC']['tau']
            self.alpha = config['SAC']['alpha']
            self.beta = config['SAC']['beta']
            self.target_entropy = config['SAC']['target_entropy_ratio']
        else:
            self.gamma = gamma
            self.tau = tau
            self.batch_size = batch_size
            self.alpha = alpha
            self.beta = beta
            self.layer1_size = layer1_size
            self.layer2_size = layer2_size

        self.update_interval = update_interval
        self.buffer_max_size = buffer_max_size
        self.scale = reward_scale
        self.lr = 0.002
        self.env = env
        self.input_dims = input_dims[0]
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir
        self.target_entropy = target_entropy_ratio  # -np.prod(action_space.shape)

        # if config is not None and 'chkpt_dir' in config["SAC"].keys():
        #     self.chkpt_dir = config['chkpt_dir']

        self.actor = Actor(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(device)
        self.critic = Critic(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(device)
        self.target_critic = Critic(self.input_dims, self.n_actions, self.layer1_size, chkpt_dir=self.chkpt_dir).to(
            device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.soft_update_target()

        # disable gradient for target critic
        # for param in self.target_critic.parameters():
        #     param.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.alpha, eps=1e-4)
        self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=self.beta, eps=1e-4)
        self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=self.beta, eps=1e-4)

        # target -> maximum entropy (same prob for each action)
        # - log ( 1 / A) = log A
        # self.target_entropy = -np.log(1.0 / action_dim) * self.target_entropy_ratio
        # self.target_entropy = np.log(action_dim) * self.target_entropy_ratio

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)

        self.memory = ReplayBuffer(self.buffer_max_size)

    def learn(self, interaction=None):
        if interaction is None:
            states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, states_, dones = interaction
            states, actions, rewards, states_, dones = [np.asarray([states]), np.asarray([actions]),
                                                        np.asarray([rewards]), np.asarray([states_]),
                                                        np.asarray([dones])]
        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        batch_transitions = states, actions, rewards, states_, dones

        weights = 1.  # default
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch_transitions, weights)
        policy_loss, entropies = self.calc_policy_loss(batch_transitions, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        update_params(self.critic_q1_optim, q1_loss)
        update_params(self.critic_q2_optim, q2_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        update_params(self.actor_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        return mean_q1, mean_q2, entropies

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)
            # next_q = (action_probs * (
            #     torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            # )).mean(dim=1).view(self.memory_batch_size, 1) # E = probs T . values

            alpha = self.log_alpha.exp()
            next_q = action_probs * (torch.min(next_q1, next_q2) - alpha * log_action_probs)
            next_q = next_q.sum(dim=1)

            target_q = rewards + (1 - dones) * self.gamma * (next_q)
            return target_q.unsqueeze(1)

        # assert rewards.shape == next_q.shape
        # return rewards + (1.0 - dones) * self.gamma * next_q

    def calc_critic_loss(self, batch, weights):
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        # errors = torch.abs(curr_q1.detach() - target_q)
        errors = None
        mean_q1, mean_q2 = None, None

        # We log means of Q to monitor training.
        # mean_q1 = curr_q1.detach().mean().item()
        # mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        # q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # with torch.no_grad():
        # Q for every actions to calculate expectations of Q.
        # q1, q2 = self.critic(states)
        # q = torch.min(q1, q2)

        q1, q2 = self.critic(states)

        alpha = self.log_alpha.exp()
        # minq = torch.min(q1, q2)
        # inside_term = alpha * log_action_probs - minq
        # policy_loss = (action_probs * inside_term).mean()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies

    def calc_entropy_loss2(self, pi_s, log_pi_s):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        alpha = self.log_alpha.exp()
        inside_term = - alpha * (log_pi_s + self.target_entropy).detach()
        entropy_loss = (pi_s * inside_term).mean()
        return entropy_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss

    def save_models(self):
        if self.chkpt_dir is not None:
            print('.... saving models ....')
            self.actor.save_checkpoint()
            self.critic.save_checkpoint()
            self.target_critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
