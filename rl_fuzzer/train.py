from rl_fuzzer.env import YouTubeEnv
from rl_fuzzer.rl_model import Actor, Critic
from rl_fuzzer.util import is_bug
import numpy as np
from data_generators.youtube_data_generator import YouTubeVideoGenerator
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import pickle


def train():
    # setup generator and env
    data_gen = YouTubeVideoGenerator()
    env = YouTubeEnv(data_gen=data_gen, num_videos=2, max_gens=10)
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
        actor.zero_grad()
        critic.zero_grad()
        env.reset()

        values = []
        rewards = []
        states = []
        actions = []
        dones = []

        # simulate iteration
        is_done = False

        while not is_done:
            state = env.embed_state()
            state = torch.from_numpy(state)
            action = actor(state).detach().numpy()
            value = critic(state).detach().numpy()
            next_state, is_done = env.step(action)
            states.append(next_state)
            actions.append(action)
            values.append(value)
            dones.append(is_done)
        
        # simulate sessions
        futures = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            for state in states:
                futures.append(executor.submit(env.get_recommendations, state))

        # get rewards
        recommendations = [future.result() for future in futures]

        # compute rewards
        bugs = set()
        for recs in recommendations:
            r = [rec for rec in recs if is_bug(rec) and rec not in bugs]
            bugs |= set(r)
            rewards.append(len(r))
            
        
        # update actor-critic
        # if discount_rewards:
        #     td_target = torch.tensor([0] + env.rewards)
        # else:
        dones = np.array([dones]).T.astype(int)
        values = np.array(values)
        rewards = np.array([rewards]).T

        states = np.mean([env.embed_state(state) for state in states], axis=1)
        states = torch.from_numpy(states)
        td_target = torch.tensor(rewards + (gamma * values) * (1 - dones), dtype=torch.float32)

        # states = np.mean(env.states, axis=1)
        # states = torch.from_numpy(states)
        # value = critic(states)
        # print(td_target.shape, value.shape)
        advantage = td_target - values
        
        # actor backprop
        actor_logit = actor(states)
        prob = F.softmax(actor_logit, 0)
        log_prob = F.log_softmax(actor_logit, 0)
        entropy = -(log_prob * prob).sum(0)
        actor_loss = (-log_prob.detach() * advantage).mean() - entropy * entropy_beta
        actor_optim.zero_grad()
        actor_loss.mean().backward()
        actor_optim.step()

        # critic backprop
        critic_logit = critic(states)
        critic_loss = F.mse_loss(critic_logit, td_target.reshape(-1))
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        torch.save(actor.state_dict(), f'rl_fuzzer/saved_weights/actor_{ep}.weights')
        torch.save(critic.state_dict(), f'rl_fuzzer/saved_weights/critic_{ep}.weights')

        
        #     # prob = F.softmax(logit, 1)
        #     # log_prob = F.log_softmax(logit, 1)
        #     # entropy = -(log_prob * prob).sum(1)
        #     # self.entropies.append(entropy)
        #     # action = prob.multinomial(num_samples=1).data
        #     # print(action.view(-1))
        #     # log_prob = log_prob.gather(1, action)
        #     # self.log_probs.append(log_prob.squeeze(1))
        #     # return action.view(-1).tolist()
        
        
        # # logs_probs = norm_dists.log_prob(300)
        # # entropy = norm_dists.entropy().mean()
        
        # actor_loss = (-advantage.detach()).mean()# - entropy*entropy_beta

        
if __name__ == '__main__':
    train()