# README
`navi_a2c_mp_ms`の概要
複数の学習データを学習することができる
## プログラム
* `myenv/env.py`
    - シミュレーション環境をOpen AIのgymのフォーマットに倣ったもの
    - `config.ini`の設定を読み込み，初期化する
    - できるだけ独立性の高いものを目指しているが, 学習データの読み込み部分に関しては修正が必要 (任意の学習データを`config.ini`を利用して読み込むなどの修正)
* `mp/envs.py`
    - マルチプロセスで動作させるためのラッパー
* `envs.py`
    - 学習のプロセスを記述
* `model.py`
    - Actor Criticの学習モデル
* `storage.py`
    - A2CのAdvanced stepを用いた行動価値関数の計算に利用する
* `brain.py`
    - 得られた報酬や状態を集計してNNに誤差逆伝播する
* `edges.py`
    - 道路網の情報管理に利用
* `main.py`
    - main関数
* `config.ini`
    - シミュレータの初期設定情報を記述
    - 観測するステップ数も記述
    - できるだけプログラムの(学習モデルとの)独立性を高めたいので，このような設定した．

## 実行例
```bash
# simulatorは事前にコンパイルが必要，nvccでコンパイルする
$ cd bin
$ python compile.py
# 初めて実行する場合
$ python main.py --save --resdir logs/N80000_advanced34 --outpufn N80000 --num-episode 100 --num-advanced-step 34
# 学習したモデルを継続して学習する場合
$ python main.py --save --resdir logs/N80000_advanced34 --outpufn N80000 --num-episode 200 --num-advanced-step 34 --checkpoint --inputfn logs/N80000_advanced34/N80000_episode100.pt --step 100
```

## 出力ファイル

* `loss_log.txt`
    - episodeごとの損失関数値を記録している
* `episode_reward.txt`
    - episodeごとの旅行時間を記録している
* `reward_log.txt`
    - 各ステップの逐次報酬と累積報酬が記録される
