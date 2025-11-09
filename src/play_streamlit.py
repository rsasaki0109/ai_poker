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
    if st.session_state.done:
        st.write("🎯 **Game not started. Click 'New Hand' to begin.**")
        if st.session_state.history:
            st.write(f"📜 **Last Game History:** {st.session_state.history}")
        return
    
    try:
        current_player = env.get_player_id()
    except AttributeError:
        st.write("❌ Error: Unable to get current player. Please start a new hand.")
        return
    
    # Display current player
    player_name = '🧑 Human (You)' if current_player == 0 else '🤖 AI'
    st.write(f"### Current Turn: {player_name}")
    
    # Get game state for current player
    state = env.get_state(current_player)
    
    # Display hand cards for human player
    if current_player == 0:
        try:
            # Extract and display hand information
            obs = state['obs']
            hand_display = get_hand_display(obs)
            st.markdown(hand_display)
            
            # Also show raw observation for debugging (in expander)
            with st.expander("🔍 Debug Info (Raw Observation)"):
                st.write(f"Raw observation: {obs}")
                
        except Exception as e:
            st.write("🃏 **Your Hand:** (Unable to display)")
            st.write(f"Debug: {e}")
    
    # Display available actions with explanations
    legal_actions = state['legal_actions']
    action_names = [env.actions[a] for a in legal_actions.keys()]
    
    st.write("🎲 **Available Actions:**")
    for i, (action_id, action_name) in enumerate(zip(legal_actions.keys(), action_names)):
        explanation = get_action_explanation(action_name)
        st.write(f"• **{action_name}**: {explanation}")
    
    # Display game history
    if st.session_state.history:
        st.write(f"📜 **Game History:** {' → '.join(st.session_state.history)}")

def get_action_explanation(action_name):
    """Get explanation for poker actions"""
    explanations = {
        'call': 'Match the current bet amount',
        'raise': 'Increase the bet amount',
        'fold': 'Give up your hand and forfeit the round',
        'check': 'Pass without betting (when no bet is required)',
        'bet': 'Place the first bet in a round'
    }
    return explanations.get(action_name.lower(), 'Perform this action')

def card_to_name(card_id):
    """Convert card ID to readable name"""
    card_names = {
        0: 'Jack ♣️',
        1: 'Queen ♥️', 
        2: 'King ♠️'
    }
    return card_names.get(int(card_id), f'Card {card_id}')

def get_hand_display(obs):
    """Extract and display hand information from observation"""
    try:
        # In Leduc Hold'em, the observation typically contains:
        # - Private card information
        # - Public card information (if revealed)
        # - Betting information
        
        if len(obs) >= 3:
            # First element is usually the private card
            private_card = obs[0]
            # Second element might be public card (if available)
            public_card = obs[1] if obs[1] != -1 else None
            
            hand_info = f"🃏 **Private Card:** {card_to_name(private_card)}"
            
            if public_card is not None:
                hand_info += f"\n🃏 **Community Card:** {card_to_name(public_card)}"
            else:
                hand_info += f"\n🃏 **Community Card:** (Not revealed yet)"
                
            return hand_info
        else:
            return f"🃏 **Hand:** {[card_to_name(x) for x in obs if x >= 0]}"
            
    except Exception as e:
        return f"🃏 **Hand:** (Unable to parse: {obs})"


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
        
        st.write("### 🎯 Choose Your Action:")
        cols = st.columns(len(legal_actions))
        for i, a in enumerate(legal_actions):
            action_name = action_names[i]
            explanation = get_action_explanation(action_name)
            button_label = f"{action_name.title()}"
            
            if cols[i].button(button_label, help=explanation, use_container_width=True):
                st.session_state.history.append(f"Human: {env.actions[a]}")
                step_env(a)
                st.rerun()

    if st.session_state.done:
        if 'env' in st.session_state:
            try:
                payoffs = st.session_state.env.get_payoffs()
                if payoffs is not None and len(payoffs) > 0:
                    st.subheader('🏆 Hand Result')
                    human_result = payoffs[0]
                    ai_result = payoffs[1]
                    
                    if human_result > ai_result:
                        st.success(f"🎉 You Won! Your payoff: +{human_result}, AI payoff: {ai_result}")
                    elif human_result < ai_result:
                        st.error(f"😔 AI Won! Your payoff: {human_result}, AI payoff: +{ai_result}")
                    else:
                        st.info(f"🤝 It's a Tie! Both payoffs: {human_result}")
            except AttributeError:
                st.subheader('🏆 Hand Result')
                st.info("Game completed. (Payoffs unavailable due to library version)")
            except Exception as e:
                st.subheader('🏆 Hand Result')
                st.warning(f"Game completed. (Error retrieving payoffs: {str(e)})")

    # Add poker rules explanation
    with st.expander("📚 Leduc Hold'em Rules & Tips"):
        st.write("""
        **Leduc Hold'em** is a simplified poker variant:
        
        🃏 **Game Setup:**
        - 6 cards total: 2 Jacks ♣️, 2 Queens ♥️, 2 Kings ♠️
        - Each player gets 1 private card
        - 1 community card is revealed after first betting round
        
        🎯 **How to Win:**
        - **Pair**: Your card + community card are the same rank (best hand!)
        - **High Card**: Higher rank wins (King ♠️ > Queen ♥️ > Jack ♣️)
        
        🃏 **Card Values:**
        - Jack ♣️ = Lowest value
        - Queen ♥️ = Medium value  
        - King ♠️ = Highest value
        
        🎲 **Actions:**
        - **Call**: Match the current bet
        - **Raise**: Increase the bet (limited raises per round)
        - **Fold**: Give up and lose your bet
        - **Check**: Pass when no bet is required
        
        💡 **Tips:**
        1. Select a model checkpoint from the sidebar
        2. Click 'New Hand' to start a new game
        3. Choose your action when it's your turn
        4. Try to read the AI's betting patterns!
        """)

if __name__ == '__main__':
    main()
