import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import rlcard

LANGUAGE_OPTIONS = ['English', '日本語']
LANGUAGE_CODE_MAP = {'English': 'en', '日本語': 'ja'}
LANGUAGE_NAME_MAP = {code: name for name, code in LANGUAGE_CODE_MAP.items()}

TEXTS = {
    'en': {
        'page_title': 'Poker RL - Leduc vs PPO',
        'app_title': "Poker RL - Leduc Hold’em (Human vs PPO)",
        'sidebar_header': 'Model',
        'language_label': 'Language',
        'model_directory': 'Model directory',
        'checkpoint_label': 'Checkpoint (.pt)',
        'load_success': 'Loaded {filename}',
        'load_fail': 'Failed to load: {error}',
        'new_hand_button': 'New Hand',
        'reset_env_button': 'Reset Env',
        'reset_info': 'Environment reset.',
        'game_not_started': "🎯 **Game not started. Click 'New Hand' to begin.**",
        'last_history': '📜 **Last Game History:** {history}',
        'current_turn': '### Current Turn: {player}',
        'player_human': '🧑 Human (You)',
        'player_ai': '🤖 AI',
        'state_error': '❌ Error: Unable to get current player. Please start a new hand.',
        'hand_label': '🃏 **Your Hand:**',
        'hand_unavailable': '(Unable to display)',
        'debug_label': '🔍 Debug Info (Raw Observation)',
        'debug_raw_observation': 'Raw observation: {obs}',
        'actions_header': '🎲 **Available Actions:**',
        'game_history': '📜 **Game History:** {history}',
        'hand_private': '🃏 **Private Card:** {card}',
        'hand_public': '🃏 **Community Card:** {card}',
        'hand_public_hidden': '🃏 **Community Card:** (Not revealed yet)',
        'hand_generic': '🃏 **Hand:** {cards}',
        'hand_parse_fail': '🃏 **Hand:** (Unable to parse: {obs})',
        'choose_action': '### 🎯 Choose Your Action:',
        'hand_result_title': '🏆 Hand Result',
        'result_win': '🎉 You Won! Your payoff: +{human}, AI payoff: {ai}',
        'result_loss': '😔 AI Won! Your payoff: {human}, AI payoff: +{ai}',
        'result_tie': "🤝 It's a Tie! Both payoffs: {human}",
        'result_unavailable': 'Game completed. (Payoffs unavailable due to library version)',
        'result_error': 'Game completed. (Error retrieving payoffs: {error})',
        'rules_title': "📚 Leduc Hold'em Rules & Tips",
        'rules_body': (
            "**Leduc Hold'em** is a simplified poker variant:\n\n"
            "🃏 **Game Setup:**\n"
            "- 6 cards total: 2 Jacks ♣️, 2 Queens ♥️, 2 Kings ♠️\n"
            "- Each player gets 1 private card\n"
            "- 1 community card is revealed after first betting round\n\n"
            "🎯 **How to Win:**\n"
            "- **Pair**: Your card + community card are the same rank (best hand!)\n"
            "- **High Card**: Higher rank wins (King ♠️ > Queen ♥️ > Jack ♣️)\n\n"
            "🃏 **Card Values:**\n"
            "- Jack ♣️ = Lowest value\n"
            "- Queen ♥️ = Medium value\n"
            "- King ♠️ = Highest value\n\n"
            "🎲 **Actions:**\n"
            "- **Call**: Match the current bet\n"
            "- **Raise**: Increase the bet (limited raises per round)\n"
            "- **Fold**: Give up and lose your bet\n"
            "- **Check**: Pass when no bet is required\n\n"
            "💡 **Tips:**\n"
            "1. Select a model checkpoint from the sidebar\n"
            "2. Click 'New Hand' to start a new game\n"
            "3. Choose your action when it's your turn\n"
            "4. Try to read the AI's betting patterns!"
        ),
        'new_hand_info': 'New hand started.',
        'ai_history_entry': 'AI: {action}',
        'human_history_entry': 'Human: {action}',
        'action_labels': {
            'call': 'Call',
            'raise': 'Raise',
            'fold': 'Fold',
            'check': 'Check',
            'bet': 'Bet'
        },
        'action_explanations': {
            'call': 'Match the current bet amount',
            'raise': 'Increase the bet amount',
            'fold': 'Give up your hand and forfeit the round',
            'check': 'Pass without betting (when no bet is required)',
            'bet': 'Place the first bet in a round'
        },
        'card_names': {
            0: 'Jack ♣️',
            1: 'Queen ♥️',
            2: 'King ♠️'
        },
        'card_default': 'Card {card}'
    },
    'ja': {
        'page_title': 'Poker RL - Leduc vs PPO (日本語)',
        'app_title': 'Poker RL - Leduc Hold’em（人間 vs PPO）',
        'sidebar_header': 'モデル',
        'language_label': '言語',
        'model_directory': 'モデルディレクトリ',
        'checkpoint_label': 'チェックポイント (.pt)',
        'load_success': '読み込み成功: {filename}',
        'load_fail': '読み込み失敗: {error}',
        'new_hand_button': '新しいハンド',
        'reset_env_button': '環境をリセット',
        'reset_info': '環境をリセットしました。',
        'game_not_started': '🎯 **ゲームが開始されていません。「新しいハンド」をクリックしてください。**',
        'last_history': '📜 **直近の履歴:** {history}',
        'current_turn': '### 現在の手番: {player}',
        'player_human': '🧑 人間（あなた）',
        'player_ai': '🤖 AI',
        'state_error': '❌ エラー: 現在のプレイヤーを取得できません。新しいハンドを開始してください。',
        'hand_label': '🃏 **あなたの手札:**',
        'hand_unavailable': '（表示できません）',
        'debug_label': '🔍 デバッグ情報（生の観測）',
        'debug_raw_observation': '生の観測: {obs}',
        'actions_header': '🎲 **選択可能なアクション:**',
        'game_history': '📜 **ゲーム履歴:** {history}',
        'hand_private': '🃏 **ホールカード:** {card}',
        'hand_public': '🃏 **コミュニティカード:** {card}',
        'hand_public_hidden': '🃏 **コミュニティカード:**（まだ公開されていません）',
        'hand_generic': '🃏 **手札:** {cards}',
        'hand_parse_fail': '🃏 **手札:**（解析できません: {obs}）',
        'choose_action': '### 🎯 アクションを選択してください:',
        'hand_result_title': '🏆 ハンド結果',
        'result_win': '🎉 勝利！ あなたのペイオフ: +{human}、AIのペイオフ: {ai}',
        'result_loss': '😔 敗北… あなたのペイオフ: {human}、AIのペイオフ: +{ai}',
        'result_tie': '🤝 引き分け！ 両者のペイオフ: {human}',
        'result_unavailable': 'ゲームは終了しました。（ライブラリのバージョンによりペイオフを取得できません）',
        'result_error': 'ゲームは終了しました。（ペイオフ取得エラー: {error}）',
        'rules_title': "📚 Leduc Hold'em のルールとヒント",
        'rules_body': (
            'Leduc Hold\'em は簡略化されたポーカーの一種です。\n\n'
            '🃏 **ゲーム準備:**\n'
            '- 合計6枚のカード: ジャック♣️ 2枚、クイーン♥️ 2枚、キング♠️ 2枚\n'
            '- 各プレイヤーはホールカードを1枚受け取ります\n'
            '- 最初のベッティングラウンド後にコミュニティカードが1枚公開されます\n\n'
            '🎯 **勝利条件:**\n'
            '- **ペア**: あなたのカードとコミュニティカードが同じランクなら最強の役です\n'
            '- **ハイカード**: ランクが高いカードが勝ちます（キング♠️ > クイーン♥️ > ジャック♣️）\n\n'
            '🃏 **カードの強さ:**\n'
            '- ジャック♣️ = 最も弱い\n'
            '- クイーン♥️ = 中くらい\n'
            '- キング♠️ = 最も強い\n\n'
            '🎲 **アクション:**\n'
            '- **コール**: 現在のベット額に合わせます\n'
            '- **レイズ**: ベット額を上げます（1ラウンド内のレイズ回数には制限があります）\n'
            '- **フォールド**: 降りてこのラウンドを放棄します\n'
            '- **チェック**: ベットが不要なときに回します\n\n'
            '💡 **遊び方のヒント:**\n'
            '1. サイドバーからモデルのチェックポイントを選択します\n'
            "2. 『新しいハンド』をクリックしてゲームを開始します\n"
            '3. 自分の手番になったらアクションを選択します\n'
            '4. AIのベットパターンを観察してみましょう！'
        ),
        'new_hand_info': '新しいハンドを開始しました。',
        'ai_history_entry': 'AI: {action}',
        'human_history_entry': '人間: {action}',
        'action_labels': {
            'call': 'コール',
            'raise': 'レイズ',
            'fold': 'フォールド',
            'check': 'チェック',
            'bet': 'ベット'
        },
        'action_explanations': {
            'call': '現在のベット額にコールします',
            'raise': 'ベット額を増やします',
            'fold': 'ハンドをフォールドしてこのラウンドを降ります',
            'check': 'ベットが不要な場合にチェックします',
            'bet': 'このラウンドで最初のベットを行います'
        },
        'card_names': {
            0: 'ジャック ♣️',
            1: 'クイーン ♥️',
            2: 'キング ♠️'
        },
        'card_default': 'カード {card}'
    }
}


def get_language():
    return st.session_state.get('lang', 'en')


def get_text(key):
    lang = get_language()
    lang_texts = TEXTS.get(lang, TEXTS['en'])
    if key in lang_texts:
        return lang_texts[key]
    return TEXTS['en'].get(key, key)


def get_action_label(action_name):
    action = action_name.lower()
    lang = get_language()
    lang_texts = TEXTS.get(lang, TEXTS['en'])
    labels = lang_texts.get('action_labels', {})
    if action in labels:
        return labels[action]
    return action_name.title()


def get_action_explanation(action_name):
    action = action_name.lower()
    lang = get_language()
    lang_texts = TEXTS.get(lang, TEXTS['en'])
    explanations = lang_texts.get('action_explanations', {})
    if action in explanations:
        return explanations[action]
    return 'Perform this action'


def get_card_name(card_id):
    try:
        idx = int(card_id)
    except (TypeError, ValueError):
        idx = card_id
    lang = get_language()
    lang_texts = TEXTS.get(lang, TEXTS['en'])
    card_names = lang_texts.get('card_names', {})
    if idx in card_names:
        return card_names[idx]
    default_fmt = lang_texts.get('card_default', TEXTS['en'].get('card_default', 'Card {card}'))
    return default_fmt.format(card=card_id)

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
    st.session_state.info = get_text('new_hand_info')


def step_env(action: int):
    env = st.session_state.env
    env.step(action)
    st.session_state.done = env.is_over()


def render_state():
    env = st.session_state.env
    if st.session_state.done:
        st.write(get_text('game_not_started'))
        if st.session_state.history:
            history_text = ' → '.join(st.session_state.history)
            st.write(get_text('last_history').format(history=history_text))
        return
    
    try:
        current_player = env.get_player_id()
    except AttributeError:
        st.write(get_text('state_error'))
        return
    
    # Display current player
    player_name = get_text('player_human') if current_player == 0 else get_text('player_ai')
    st.write(get_text('current_turn').format(player=player_name))
    
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
            with st.expander(get_text('debug_label')):
                st.write(get_text('debug_raw_observation').format(obs=obs))
                
        except Exception as e:
            st.write(f"{get_text('hand_label')} {get_text('hand_unavailable')}")
            st.write(f"Debug: {e}")
    
    # Display available actions with explanations
    legal_actions_dict = state['legal_actions']
    action_keys = list(legal_actions_dict.keys())
    action_names = [env.actions[a] for a in action_keys]

    st.write(get_text('actions_header'))
    for action_id, action_name in zip(action_keys, action_names):
        explanation = get_action_explanation(action_name)
        display_name = get_action_label(action_name)
        st.write(f"• **{display_name}**: {explanation}")
    
    # Display game history
    if st.session_state.history:
        history_text = ' → '.join(st.session_state.history)
        st.write(get_text('game_history').format(history=history_text))

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

            lines = [
                get_text('hand_private').format(card=get_card_name(private_card))
            ]

            if public_card is not None:
                lines.append(get_text('hand_public').format(card=get_card_name(public_card)))
            else:
                lines.append(get_text('hand_public_hidden'))

            return "\n".join(lines)
        else:
            cards = [get_card_name(x) for x in obs if isinstance(x, (int, np.integer)) and x >= 0]
            cards_str = ', '.join(cards) if cards else str(obs)
            return get_text('hand_generic').format(cards=cards_str)

    except Exception:
        return get_text('hand_parse_fail').format(obs=obs)


def main():
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'

    st.set_page_config(page_title=get_text('page_title'), page_icon='🂡')
    st.title(get_text('app_title'))

    init_session()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.sidebar.header(get_text('sidebar_header'))

    current_lang = get_language()
    current_option = LANGUAGE_NAME_MAP.get(current_lang, LANGUAGE_OPTIONS[0])
    language_label = TEXTS[current_lang if current_lang in TEXTS else 'en'].get('language_label', 'Language')
    selected_option = st.sidebar.selectbox(language_label, LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index(current_option))
    selected_lang = LANGUAGE_CODE_MAP.get(selected_option, 'en')
    if selected_lang != current_lang:
        st.session_state.lang = selected_lang
        st.rerun()

    model_dir = st.sidebar.text_input(get_text('model_directory'), value='models/ppo_leduc')
    ckpts = list_checkpoints(model_dir)
    default_idx = max(0, len(ckpts) - 1)
    ckpt_label = get_text('checkpoint_label')
    ckpt = st.sidebar.selectbox(ckpt_label, ckpts, index=default_idx if ckpts else 0) if ckpts else None

    model = None
    if ckpt:
        try:
            model = load_model(st.session_state.env, ckpt, device)
            st.sidebar.success(get_text('load_success').format(filename=os.path.basename(ckpt)))
        except Exception as e:
            st.sidebar.error(get_text('load_fail').format(error=e))

    col1, col2 = st.columns(2)
    with col1:
        if st.button(get_text('new_hand_button')):
            start_new_hand()
    with col2:
        if st.button(get_text('reset_env_button')):
            st.session_state.env = rlcard.make('leduc-holdem')
            st.session_state.done = True
            st.session_state.history = []
            st.session_state.info = get_text('reset_info')

    if st.session_state.info:
        st.info(st.session_state.info)

    # Auto-play AI turns until it's human's turn or the hand is done
    if not st.session_state.done and model is not None:
        env = st.session_state.env
        loop_safety = 0
        while not st.session_state.done and env.get_player_id() == 1 and loop_safety < 20:
            s = env.get_state(1)
            obs = s['obs']
            legal_actions = list(s['legal_actions'].keys())
            a = policy_action(model, obs, legal_actions, device)
            action_label = get_action_label(env.actions[a])
            st.session_state.history.append(get_text('ai_history_entry').format(action=action_label))
            step_env(a)
            loop_safety += 1

    render_state()

    # Human controls
    env = st.session_state.env
    if not st.session_state.done and env.get_player_id() == 0:
        s = env.get_state(0)
        legal_actions = list(s['legal_actions'].keys())
        action_names = [env.actions[a] for a in legal_actions]
        
        st.write(get_text('choose_action'))
        cols = st.columns(len(legal_actions))
        for i, a in enumerate(legal_actions):
            action_name = action_names[i]
            explanation = get_action_explanation(action_name)
            button_label = get_action_label(action_name)
            
            if cols[i].button(button_label, help=explanation, use_container_width=True):
                st.session_state.history.append(get_text('human_history_entry').format(action=button_label))
                step_env(a)
                st.rerun()

    if st.session_state.done:
        if 'env' in st.session_state:
            try:
                payoffs = st.session_state.env.get_payoffs()
                if payoffs is not None and len(payoffs) > 0:
                    st.subheader(get_text('hand_result_title'))
                    human_result = payoffs[0]
                    ai_result = payoffs[1]
                    
                    if human_result > ai_result:
                        st.success(get_text('result_win').format(human=human_result, ai=ai_result))
                    elif human_result < ai_result:
                        st.error(get_text('result_loss').format(human=human_result, ai=ai_result))
                    else:
                        st.info(get_text('result_tie').format(human=human_result))
            except AttributeError:
                st.subheader(get_text('hand_result_title'))
                st.info(get_text('result_unavailable'))
            except Exception as e:
                st.subheader(get_text('hand_result_title'))
                st.warning(get_text('result_error').format(error=str(e)))

    # Add poker rules explanation
    with st.expander(get_text('rules_title')):
        st.write(get_text('rules_body'))

if __name__ == '__main__':
    main()
