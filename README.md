# Local-LLM Lecture Transcriber Bot

大学の講義録音を文字起こしし、誤認識を校正したうえで要約する Discord ボットです。音声認識と文章処理の推論は、ボットを動かしているマシン上の faster-whisper と Ollama で行います。

## データの境界

音声ファイルは利用者が Discord にアップロードしたものです。ボットはそれを受け取り、校正済みテキストと要約を Discord へ返します。入出力は Discord のサーバーを経由します。

推論そのものは OpenAI などの外部の文字起こし API や要約 API へ送りません。ただし次の通信は発生します。

- Discord ゲートウェイへの常時接続と、結果ファイルの送信
- 初回起動時の `large-v3` モデル取得（取得後の認識処理はローカル）
- 事前に行う `ollama pull command-r`（取得後の校正と要約はローカルの Ollama）

講義音声には他人の発言や個人情報が含まれることがあります。`archives/` と `.env` は Git の管理外です。実在の Bot トークンと Discord ユーザー ID は、コードにもこの README にも書かないでください。

## 機能

1. **ローカル推論**
   - 音声認識は `faster-whisper` の `large-v3`、校正と要約はローカルの Ollama（`command-r`）です。
2. **文字起こしの校正**
   - 要約の前に、温度 0.1 の校正を挟みます。対象は先頭 4000 文字です。それを超えた部分は未校正のまま後ろへ連結します。
3. **許可ユーザーだけの受付**
   - 環境変数 `ALLOWED_USER_IDS` に含まれる数字のユーザー ID だけを処理します。未設定または空のときは全員を拒否します。ID はコードに埋め込みません。

## システム構成

- **言語**: Python 3.10 以上（`requests` と `python-dotenv` の要件。この整備時の確認は Python 3.12）
- **インターフェース**: Discord API（`discord.py`）
- **音声認識**: `faster-whisper`（`large-v3`）
- **文章処理**: Ollama（`command-r`、`http://localhost:11434`）
- **依存関係**: `requirements.txt`

### 処理フロー

1. 許可されたユーザーが音声ファイルを Discord に添付する
2. 拡張子、サイズ（25 MiB 以下）、ファイルヘッダを確認し、問題があれば破棄する
3. `archives/` 以下へ保存する。保存先のパスは Discord へ返さない
4. `faster-whisper` が音声をテキスト化する。この処理はワーカースレッドで行い、Discord のイベントループを止めない
5. Ollama が先頭 4000 文字を校正する。HTTP タイムアウトは 180 秒
6. 校正済みテキストをローカルに保存し、Discord へファイルとして返す
7. Ollama が先頭 6000 文字から要約を作る
8. 要約が Discord の 2000 文字制限を超える場合は複数メッセージに分割して返す

失敗時に Discord へ返すのは状況を示す短い文だけです。例外の全文とローカルパスは、ボットを動かしているマシンの標準出力に残します。

文字起こし本文は、LLM への命令文とは別の区切り（`<<<UNTRUSTED_TRANSCRIPT>>>` から `<<<END_UNTRUSTED_TRANSCRIPT>>>`）の内側に入れます。プロンプトでは、内側をデータとして扱い、内側の指示には従わないよう指定しています。

## Discord 側で必要な設定

1. [Discord Developer Portal](https://discord.com/developers/applications) でアプリケーションと Bot を作る
2. Bot の **Privileged Gateway Intents** で **Message Content Intent** を有効にする。サーバー内の添付ファイルを受け取るために必要です
3. OAuth2 のスコープ `bot` でサーバーへ招待する。必要な権限は、チャンネルの閲覧、メッセージの送信、ファイルの添付、メッセージ履歴の閲覧です
4. 自分の Discord ユーザー ID を確認する。開発者モードを有効にし、ユーザーを右クリックして ID をコピーします
5. `.env.example` を `.env` にコピーし、次をそのマシンだけに書く
   - `DISCORD_BOT_TOKEN`: Bot トークン
   - `ALLOWED_USER_IDS`: 許可するユーザー ID。複数のときはカンマ区切り。数字以外は無視されます

`.env` はコミットしないでください。許可 ID を空のまま起動すると、ボットは誰の添付も処理しません。

## ローカル側の準備

Ollama を同じマシンで起動し、モデルを取得します。

```bash
ollama pull command-r
```

Python の依存パッケージを入れます。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

GPU が使えない場合、Whisper は CPU（`int8`）へフォールバックします。`large-v3` の CPU 推論は講義の長さによっては時間がかかります。推論中も Discord へのハートビートはイベントループ側で続きます。同時に複数の音声が来た場合、Whisper と Ollama の呼び出しはロックで直列化します。

## 制限

| 項目 | 実装 |
| --- | --- |
| 添付 | メッセージ内の先頭ファイルだけを見る |
| 形式 | `.mp3` `.m4a` `.wav` `.ogg` `.webm`。拡張子とヘッダの両方が必要 |
| サイズ | 1 バイト以上、25 MiB 以下 |
| 校正 | 先頭 4000 文字。以降は未校正のテキストを連結 |
| 要約 | 校正結果の先頭 6000 文字 |
| Ollama | `timeout=180` 秒。モデル名は `command-r` |
| Discord の返信 | 2000 文字ごとに分割 |
| 許可リスト | `ALLOWED_USER_IDS`。空なら全員拒否 |
| メンション | LLM の出力で `@everyone` などが飛ばないよう、メンションは無効 |

ヘッダが一般的な形式と異なる音声は拒否します。校正は文の途中で 4000 文字に切れることがあります。

## ディレクトリ構成

```text
.
├── bot.py            # アプリケーション
├── requirements.txt  # 依存パッケージ（バージョン固定）
├── .env.example      # 環境変数の名前だけを示す見本
├── .gitignore
├── .env              # Git 管理外。トークンと許可ユーザー ID
└── archives/         # Git 管理外。音声と文字起こし
```
