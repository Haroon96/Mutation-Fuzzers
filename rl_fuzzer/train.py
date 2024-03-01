from rl_fuzzer.env import YouTubeEnv
from rl_fuzzer.rl_model import Actor, Critic
import numpy as np
from data_generators.youtube_data_generator import YouTubeVideoGenerator
import torch
import torch.nn.functional as F
import pickle


def train():
    # setup generator and env
    data_gen = YouTubeVideoGenerator()
    env = YouTubeEnv(data_gen=data_gen, num_videos=2, max_gens=2)
    actor = Actor()
    critic = Critic()
    
    discount_rewards = True
    gamma = 0.9
    entropy_beta = 0
    actor_lr=4e-4
    critic_lr=4e-3
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    # run episodes
    for ep in range(500):
        
        # run episodes
        # while not env.is_done():
        #     state = np.mean(env.state, axis=0)
        #     state = torch.from_numpy(state)
        #     action = actor(state).detach()
        #     env.step(action)
        
        with open('rl_fuzzer/data/sample.pickle', 'rb') as f:
            env.states, env.state_videos, env.rewards, env.actions, env.next_states = pickle.load(f)
            
        # update actor-critic
        if discount_rewards:
            td_target = torch.tensor([0] + env.rewards)
        else:
            next_states = np.mean(env.next_states, axis=1)
            next_states = torch.from_numpy(next_states)
            td_target = env.rewards + gamma*critic(next_states)*(1-env.dones)
        states = np.mean(env.states, axis=1)
        states = torch.from_numpy(states)
        value = critic(states)
        print(td_target.shape, value.shape)
        advantage = td_target - value[0]
        
        # actor
        states = np.mean(env.states, axis=1)
        states = torch.from_numpy(states)
        logit = actor(states)
        prob = F.softmax(logit, 0)
        log_prob = F.log_softmax(logit, 0)
        entropy = -(log_prob * prob).sum(0)
        
        
            # prob = F.softmax(logit, 1)
            # log_prob = F.log_softmax(logit, 1)
            # entropy = -(log_prob * prob).sum(1)
            # self.entropies.append(entropy)
            # action = prob.multinomial(num_samples=1).data
            # print(action.view(-1))
            # log_prob = log_prob.gather(1, action)
            # self.log_probs.append(log_prob.squeeze(1))
            # return action.view(-1).tolist()
        
        
        # logs_probs = norm_dists.log_prob(300)
        # entropy = norm_dists.entropy().mean()
        
        actor_loss = (-log_prob*advantage.detach()).mean() - entropy*entropy_beta
        # actor_loss = (-advantage.detach()).mean()# - entropy*entropy_beta
        actor_optim.zero_grad()
        actor_loss.backward()
        
        # clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        critic_optim.zero_grad()
        critic_loss.backward()
        # clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        critic_optim.step()
        
if __name__ == '__main__':
    train()