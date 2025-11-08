import os
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import rlcard

# Simple PPO for RLCard Leduc Hold'em (self-play, shared policy)

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_iters: int = 4
    minibatch_size: int = 2048
    steps_per_iter: int = 4096
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 128
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: torch.device):
        self.device = device
        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer_pi = optim.Adam(self.model.pi.parameters(), lr=cfg.pi_lr)
        self.optimizer_v = optim.Adam(self.model.v.parameters(), lr=cfg.vf_lr)
        self.cfg = cfg
        self.act_dim = act_dim

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, legal_actions: List[int]):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(x)
        logits = logits[0].clone()
        # Mask illegal actions by putting very negative logits
        mask = torch.full((self.act_dim,), float('-inf'), device=self.device)
        mask[legal_actions] = 0.0
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item()), probs.detach().cpu().numpy()

    def update(self, batch):
        obs = torch.tensor(batch['obs'], dtype=torch.float32, device=self.device)
        act = torch.tensor(batch['act'], dtype=torch.long, device=self.device)
        adv = torch.tensor(batch['adv'], dtype=torch.float32, device=self.device)
        ret = torch.tensor(batch['ret'], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(batch['logp'], dtype=torch.float32, device=self.device)
        legal_mask = torch.tensor(batch['legal_mask'], dtype=torch.float32, device=self.device)

        for _ in range(self.cfg.train_iters):
            # Policy loss
            logits, value = self.model(obs)
            # Apply mask
            masked_logits = logits + (legal_mask + 1e-10).log()
            log_probs = F.log_softmax(masked_logits, dim=-1)
            logp = log_probs.gather(1, act.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
            pi_loss = -(torch.min(ratio * adv, clipped)).mean()
            # Entropy bonus over legal actions
            probs = F.softmax(masked_logits, dim=-1)
            entropy = -(probs * log_probs).sum(-1).mean()

            # Value loss
            v_loss = F.mse_loss(value, ret)

            # Optimize
            self.optimizer_pi.zero_grad()
            (pi_loss - self.cfg.entropy_coef * entropy).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer_pi.step()

            self.optimizer_v.zero_grad()
            (self.cfg.vf_coef * v_loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer_v.step()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state)


def compute_gae(rollout: List[dict], cfg: PPOConfig) -> Tuple[np.ndarray, np.ndarray]:
    # rollout: list of steps for the learning player only
    values = np.array([step['value'] for step in rollout] + [0.0], dtype=np.float32)
    rewards = np.array([step['reward'] for step in rollout], dtype=np.float32)
    dones = np.array([step['done'] for step in rollout], dtype=np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + cfg.gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


def collect_trajectories(env, agent: PPOAgent, steps_target: int, device: torch.device, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Use agent dimensions to avoid RLCard API differences
    act_dim = agent.act_dim

    batch = {
        'obs': [],
        'act': [],
        'logp': [],
        'val': [],
        'rew': [],
        'done': [],
        'legal_mask': [],
    }

    steps = 0
    while steps < steps_target:
        env.reset()
        # Per-player buffers to attribute rewards correctly
        per_player_buffers = {0: [], 1: []}
        while not env.is_over():
            current_player = env.get_player_id()
            s = env.get_state(current_player)
            obs = s['obs']
            legal_actions = list(s['legal_actions'].keys())
            action, logp, value, probs = agent.select_action(obs, legal_actions)

            legal_mask = np.zeros(act_dim, dtype=np.float32)
            legal_mask[legal_actions] = 1.0

            # store step for this player
            per_player_buffers[current_player].append({
                'obs': obs,
                'act': action,
                'logp': logp,
                'value': value,
                'legal_mask': legal_mask,
            })

            env.step(action)

        # Episode ended
        payoffs = env.get_payoffs()  # length = num_players
        # attribute terminal rewards to each player's steps
        for p in [0, 1]:
            if len(per_player_buffers[p]) == 0:
                continue
            # Build rollout for player p
            rollout = []
            for i, st in enumerate(per_player_buffers[p]):
                rollout.append({
                    'value': st['value'],
                    'reward': payoffs[p] if i == len(per_player_buffers[p]) - 1 else 0.0,
                    'done': 1.0 if i == len(per_player_buffers[p]) - 1 else 0.0,
                })
            adv, ret = compute_gae(rollout, agent.cfg)
            for i, st in enumerate(per_player_buffers[p]):
                batch['obs'].append(st['obs'])
                batch['act'].append(st['act'])
                batch['logp'].append(st['logp'])
                batch['val'].append(rollout[i]['value'])
                batch['rew'].append(rollout[i]['reward'])
                batch['done'].append(rollout[i]['done'])
                batch['legal_mask'].append(st['legal_mask'])
                steps += 1
                if steps >= steps_target:
                    break
        # continue episodes until enough steps are collected
    # Recompute advantage/return using stored values/rew/done across the whole batch
    rollout = []
    for i in range(len(batch['obs'])):
        rollout.append({'value': batch['val'][i], 'reward': batch['rew'][i], 'done': batch['done'][i]})
    adv, ret = compute_gae(rollout, agent.cfg)
    batch_out = {
        'obs': np.array(batch['obs'], dtype=np.float32),
        'act': np.array(batch['act'], dtype=np.int64),
        'logp': np.array(batch['logp'], dtype=np.float32),
        'adv': (adv - adv.mean()) / (adv.std() + 1e-8),
        'ret': ret.astype(np.float32),
        'legal_mask': np.array(batch['legal_mask'], dtype=np.float32),
    }
    return batch_out


def evaluate(env, agent: PPOAgent, episodes: int = 200) -> float:
    wins = 0
    for _ in range(episodes):
        env.reset()
        while not env.is_over():
            current_player = env.get_player_id()
            s = env.get_state(current_player)
            obs = s['obs']
            legal_actions = list(s['legal_actions'].keys())
            if current_player == 0:
                action, _, _, _ = agent.select_action(obs, legal_actions)
            else:
                # opponent: random baseline
                action = np.random.choice(legal_actions)
            env.step(action)
        payoffs = env.get_payoffs()
        if payoffs[0] > payoffs[1]:
            wins += 1
    return wins / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=40000)
    parser.add_argument('--steps-per-iter', type=int, default=4096)
    parser.add_argument('--save-dir', type=str, default='models/ppo_leduc')
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = rlcard.make('leduc-holdem')
    # Robust obs/action dimensions across RLCard versions
    ss0 = env.state_shape[0]
    obs_dim = ss0 if isinstance(ss0, int) else ss0[0]
    act_dim = len(env.actions)

    cfg = PPOConfig(steps_per_iter=args.steps_per_iter, seed=args.seed)
    agent = PPOAgent(obs_dim, act_dim, cfg, device)

    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = args.save_dir

    steps = 0
    iter_idx = 0
    best_winrate = 0.0
    start = time.time()
    while steps < args.total_steps:
        batch = collect_trajectories(env, agent, cfg.steps_per_iter, device, seed=cfg.seed + iter_idx)
        agent.update(batch)
        steps += cfg.steps_per_iter
        iter_idx += 1
        if steps % args.eval_interval == 0 or steps >= args.total_steps:
            winrate = evaluate(env, agent, episodes=200)
            elapsed = time.time() - start
            print(f"Steps={steps}  WinRateVsRandom={winrate:.3f}  Elapsed={elapsed/60:.1f}m")
            # save checkpoint
            ckpt = os.path.join(save_dir, f'policy_steps{steps}.pt')
            agent.save(ckpt)
            if winrate > best_winrate:
                best_winrate = winrate
                agent.save(os.path.join(save_dir, 'best.pt'))

    # final save
    agent.save(os.path.join(save_dir, 'final.pt'))

if __name__ == '__main__':
    main()
