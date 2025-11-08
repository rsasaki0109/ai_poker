import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import rlcard

# Minimal ActorCritic matching train_ppo.py
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

@torch.no_grad()
def policy_action(model: ActorCritic, obs: np.ndarray, legal_actions, device):
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(x)
    logits = logits[0].clone()
    act_dim = logits.shape[-1]
    mask = torch.full((act_dim,), float('-inf'), device=device)
    mask[legal_actions] = 0.0
    masked_logits = logits + mask
    probs = F.softmax(masked_logits, dim=-1)
    a = torch.multinomial(probs, 1)[0].item()
    return int(a)


def list_checkpoints(model_dir: str):
    if not os.path.isdir(model_dir):
        return []
    files = glob.glob(os.path.join(model_dir, '*.pt'))
    files.sort()
    return files


def load_model(env, ckpt_path: str, device):
    ss0 = env.state_shape[0]
    obs_dim = ss0 if isinstance(ss0, int) else ss0[0]
    act_dim = len(env.actions)
    model = ActorCritic(obs_dim, act_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def init_session():
    if 'env' not in st.session_state:
        st.session_state.env = rlcard.make('leduc-holdem')
        st.session_state.done = True
        st.session_state.history = []
        st.session_state.info = ''


def start_new_hand():
    env = st.session_state.env
    env.reset()
    st.session_state.done = False
    st.session_state.history = []
    st.session_state.info = 'New hand started.'


def step_env(action: int):
    env = st.session_state.env
    env.step(action)
    st.session_state.done = env.is_over()


def render_state():
    env = st.session_state.env
    current_player = env.get_player_id()
    st.write(f"Current player: {'Human(0)' if current_player == 0 else 'AI(1)'}")
    st.write(f"Legal actions: {env.get_state(current_player)['legal_actions']}")
    st.write(f"Action names: {[env.actions[a] for a in env.get_state(current_player)['legal_actions'].keys()]}")
    st.write(f"History: {st.session_state.history}")


def main():
    st.set_page_config(page_title='Poker RL - Leduc vs PPO', page_icon='🂡')
    st.title('Poker RL - Leduc Hold’em (Human vs PPO)')

    init_session()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.sidebar.header('Model')
    model_dir = st.sidebar.text_input('Model directory', value='models/ppo_leduc')
    ckpts = list_checkpoints(model_dir)
    default_idx = max(0, len(ckpts) - 1)
    ckpt = st.sidebar.selectbox('Checkpoint (.pt)', ckpts, index=default_idx if ckpts else 0) if ckpts else None

    model = None
    if ckpt:
        try:
            model = load_model(st.session_state.env, ckpt, device)
            st.sidebar.success(f'Loaded {os.path.basename(ckpt)}')
        except Exception as e:
            st.sidebar.error(f'Failed to load: {e}')

    col1, col2 = st.columns(2)
    with col1:
        if st.button('New Hand'):
            start_new_hand()
    with col2:
        if st.button('Reset Env'):
            st.session_state.env = rlcard.make('leduc-holdem')
            st.session_state.done = True
            st.session_state.history = []
            st.session_state.info = 'Environment reset.'

    # Auto-play AI turns until it's human's turn or the hand is done
    if not st.session_state.done and model is not None:
        env = st.session_state.env
        loop_safety = 0
        while not st.session_state.done and env.get_player_id() == 1 and loop_safety < 20:
            s = env.get_state(1)
            obs = s['obs']
            legal_actions = list(s['legal_actions'].keys())
            a = policy_action(model, obs, legal_actions, device)
            st.session_state.history.append(f"AI: {env.actions[a]}")
            step_env(a)
            loop_safety += 1

    render_state()

    # Human controls
    env = st.session_state.env
    if not st.session_state.done and env.get_player_id() == 0:
        s = env.get_state(0)
        legal_actions = list(s['legal_actions'].keys())
        action_names = [env.actions[a] for a in legal_actions]
        cols = st.columns(len(legal_actions))
        for i, a in enumerate(legal_actions):
            if cols[i].button(action_names[i]):
                st.session_state.history.append(f"Human: {env.actions[a]}")
                step_env(a)
                st.experimental_rerun()

    if st.session_state.done:
        if 'env' in st.session_state:
            payoffs = st.session_state.env.get_payoffs()
            if payoffs:
                st.subheader('Hand Result')
                st.write(f"Payoffs: Human(0)={payoffs[0]}, AI(1)={payoffs[1]}")

    st.caption('Tips: 1) 学習後に最新checkpointを選択 2) New Handで開始 3) 人間(0)の番でアクションボタン表示')

if __name__ == '__main__':
    main()
