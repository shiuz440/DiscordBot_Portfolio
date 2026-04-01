import discord
import os
import requests
import asyncio
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import datetime
import re

# --- 設定 ---
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "command-r"
WHISPER_MODEL_SIZE = "large-v3"

# ==========================================
# 👇 ここに自分のIDを貼り直してください！
# ==========================================
ALLOWED_USER_IDS = {"YOUR_DISCORD_ID_HERE"} 

# 安全対策
MAX_TRANSCRIPT_CHARS = 6000 

# --- Discord設定 ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# グローバル変数
whisper_model = None

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    print(f"🔒 Security: Allowed User IDs = {ALLOWED_USER_IDS}")
    print("🚀 Bot is ready and listening!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if str(message.author.id) not in ALLOWED_USER_IDS:
        print(f"⛔ Unauthorized access attempt from {message.author.name} ({message.author.id})")
        return

    if message.attachments:
        attachment = message.attachments[0]
        if attachment.filename.endswith(('.mp3', '.m4a', '.wav', '.ogg', '.webm')):
            
            guild_name = sanitize_filename(message.guild.name) if message.guild else "DirectMessage"
            channel_name = sanitize_filename(message.channel.name) if hasattr(message.channel, 'name') else "DM"
            save_dir = f"./archives/{guild_name}/{channel_name}"
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_base = f"{timestamp}_{sanitize_filename(attachment.filename)}"
            audio_path = os.path.join(save_dir, file_base)
            txt_path = audio_path + ".txt"

            await message.reply(f"📥 受信。保存先: `{save_dir}`\nサーバーで解析中...")
            await attachment.save(audio_path)
            
            try:
                # 1. 音声認識 (Whisper)
                segments, info = whisper_model.transcribe(audio_path, beam_size=5)
                raw_text = " ".join([segment.text for segment in segments])
                print(f"文字起こし完了(生データ): {len(raw_text)}文字")
                
                # 2. AI校正 (Ollama) - ここが新機能！
                await message.channel.send("🔧 専門用語の聞き間違いをAI校正中...")
                
                # 長すぎる場合は分割せず、冒頭だけ校正して残りはそのまま（エラー回避のため）
                # ※本来は分割処理が必要ですが、まずはシンプルに実装します
                input_for_correction = raw_text[:4000] 
                corrected_text = correct_transcription(input_for_correction)
                
                # 4000文字以降があればそのままくっつける
                if len(raw_text) > 4000:
                    corrected_text += raw_text[4000:]
                
                # 校正済みテキストを保存
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(corrected_text)
                
                await message.channel.send(f"📄 文字起こし結果 (校正済み)", file=discord.File(txt_path))

                # 3. 要約 (Ollama)
                await message.channel.send("🧠 要約を作成中...")
                
                summary = generate_summary(corrected_text[:MAX_TRANSCRIPT_CHARS])
                await message.reply(f"📝 **講義要約**\n{summary}")

            except Exception as e:
                await message.reply(f"❌ エラー: {e}")
                print(f"Error: {e}")

# --- 聞き間違いを直す関数 ---
def correct_transcription(text):
    prompt = f"""
    [指令]
    以下のテキストは音声認識の結果です。「ピカルセンキング→クリティカルシンキング」のような
    文脈に合わない同音異義語や、不自然な専門用語の変換ミスを修正してください。
    
    [ルール]
    ・**修正後の全文**のみを出力すること。
    ・要約は絶対にしないこと。
    ・挨拶や「修正しました」等の前置きは不要。
    
    [対象テキスト]
    {text}
    """
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1} # 少しだけ柔軟性を持たせて修正させる
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return text # エラーなら元のテキストを返す
    except:
        return text

# --- 事実厳守の要約関数 ---
def generate_summary(text):
    prompt = f"""
    [指令]
    あなたは厳格なアーカイブ記録係です。
    以下の「講義の文字起こし」を、**そこに含まれる情報だけ**を使って要約してください。
    
    [禁止事項]
    ・あなたの知識や一般的な説明を付け加えること
    ・「この授業では～を学びます」のような予測を書くこと
    ・元の発言にない単語や文脈を創作すること
    
    [対象テキスト]
    {text}
    """
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection Error: {e}"

if __name__ == "__main__":
    print("⏳ Whisperモデルを読み込んでいます... (GPU)")
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
        print("✅ GPU Load Success (CUDA)")
    except Exception as e:
        print(f"⚠️ GPU失敗、CPUで動かします: {e}")
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    
    print("🔗 Discordに接続中...")
    client.run(TOKEN)