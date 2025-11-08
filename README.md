# Poker RL (Leduc Hold’em + PPO)

このリポジトリは、Leduc Hold’em を題材に PPO（自己対戦）で学習し、学習済みエージェントと人間がブラウザから対戦できる最小構成です。

## 構成
- `requirements.txt` 依存関係
- `src/train_ppo.py` PPO自己対戦での学習スクリプト（チェックポイント保存）
- `src/play_streamlit.py` 学習済みモデルと人間が対戦できるUI
- `models/` 学習済みモデル保存先（自動作成）

## セットアップ
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 学習
短時間の試し学習（デフォルト設定）
```
python src/train_ppo.py --total-steps 20000 --eval-interval 2000 --save-dir models/ppo_leduc
```
学習時間を伸ばすほど強くなります。

## 対戦（Streamlit）
別ターミナルでUIを起動:
```
streamlit run src/play_streamlit.py
```
ブラウザが開かない場合は http://localhost:8501 を開いてください。サイドバーの「Model directory」から `models/ppo_leduc` を選択（または入力）してください。

## 今後の拡張
- PPO自己対戦版の追加
- Limit / No-Limit Texas Hold’em への拡張
- 評価用リーグ戦・ハイパラ自動探索
