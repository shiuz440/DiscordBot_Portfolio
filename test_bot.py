import asyncio
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path

import requests

import bot


def wav_bytes(marker: bytes) -> bytes:
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + (b"\x00" * 32) + marker


class Segment:
    def __init__(self, text: str):
        self.text = text


class Channel:
    def __init__(self):
        self.name = "lecture"
        self.sent = []

    async def send(self, content=None, **kwargs):
        self.sent.append((content, kwargs))


class Guild:
    name = "Campus"


class Author:
    def __init__(self, user_id: int):
        self.id = user_id


class Attachment:
    def __init__(self, filename: str, payload: bytes, save_error: BaseException | None = None):
        self.filename = filename
        self.size = len(payload)
        self.payload = payload
        self.save_error = save_error

    async def save(self, path: str) -> None:
        Path(path).write_bytes(self.payload)
        if self.save_error is not None:
            raise self.save_error


class Message:
    def __init__(self, user_id: int, attachment: Attachment, channel: Channel | None = None):
        self.author = Author(user_id)
        self.channel = channel or Channel()
        self.guild = Guild()
        self.attachments = [attachment]
        self.replies = []

    async def reply(self, content=None, **kwargs):
        self.replies.append(content)


def public_text(message: Message) -> str:
    parts = list(message.replies)
    parts.extend(content for content, _kwargs in message.channel.sent if content)
    return "\n".join(parts)


def files_under(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


class BotBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archives = self.root / "archives"
        old_dir = self.archives / "Old" / "ch"
        old_dir.mkdir(parents=True)
        self.old_file = old_dir / "kept.wav"
        self.old_file.write_bytes(b"keep-existing-archive")

        self._archives = bot.ARCHIVES_DIR
        self._allowed = set(bot.ALLOWED_USER_IDS)
        self._model = bot.whisper_model
        self._correct = bot.correct_transcription
        self._summary = bot.generate_summary
        self._post = bot.requests.post
        self._mkdtemp = bot.tempfile.mkdtemp
        self._lock = bot.whisper_model_lock
        self.job_dirs: list[str] = []

        bot.ARCHIVES_DIR = str(self.archives)
        bot.ALLOWED_USER_IDS = {"42"}
        bot.correct_transcription = lambda text: (text + "|ok", True)
        bot.generate_summary = lambda text: "要約"
        bot.whisper_model = self._reading_model()

        def tracking_mkdtemp(prefix: str = "tmp", dir=None):
            path = self._mkdtemp(prefix=prefix, dir=str(self.root))
            self.job_dirs.append(path)
            return path

        bot.tempfile.mkdtemp = tracking_mkdtemp

    def tearDown(self):
        bot.ARCHIVES_DIR = self._archives
        bot.ALLOWED_USER_IDS = self._allowed
        bot.whisper_model = self._model
        bot.correct_transcription = self._correct
        bot.generate_summary = self._summary
        bot.requests.post = self._post
        bot.tempfile.mkdtemp = self._mkdtemp
        bot.whisper_model_lock = self._lock
        self.tmp.cleanup()

    def _reading_model(self):
        parent = self

        class Model:
            def transcribe(self, audio_path, beam_size=5):
                data = Path(audio_path).read_bytes()
                marker = data[-4:].decode("ascii")
                parent.job_dirs  # keep a reference so the closure stays explicit
                return [Segment(marker)], None

        return Model()

    def _assert_only_old_archive(self):
        self.assertEqual(files_under(self.root), {Path("archives/Old/ch/kept.wav")})
        self.assertEqual(self.old_file.read_bytes(), b"keep-existing-archive")
        for job_dir in self.job_dirs:
            self.assertFalse(Path(job_dir).exists(), job_dir)

    def test_success_keeps_archive_and_removes_temp(self):
        message = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))
        asyncio.run(bot.on_message(message))
        archived = [path for path in self.archives.rglob("*") if path.is_file() and path != self.old_file]
        names = sorted(path.name for path in archived)
        self.assertEqual(len(archived), 2)
        self.assertTrue(any(name.endswith(".wav") for name in names))
        self.assertTrue(any(name.endswith(".wav.txt") for name in names))
        text_file = next(path for path in archived if path.suffix == ".txt")
        self.assertEqual(text_file.read_text(encoding="utf-8"), "AAAA|ok")
        self.assertIn("校正済み", public_text(message))
        self.assertNotIn("未校正", public_text(message))
        self.assertEqual(self.old_file.read_bytes(), b"keep-existing-archive")
        for job_dir in self.job_dirs:
            self.assertFalse(Path(job_dir).exists())

    def test_bad_header_leaves_no_temp_or_new_archive(self):
        message = Message(42, Attachment("lecture.mp3", b"this is not audio content!!"))
        asyncio.run(bot.on_message(message))
        self.assertIn("内容を確認できなかった", public_text(message))
        self.assertNotIn(str(self.root), public_text(message))
        self._assert_only_old_archive()

    def test_save_failure_cleans_temp(self):
        attachment = Attachment("lecture.wav", wav_bytes(b"AAAA"), save_error=OSError("disk"))
        message = Message(42, attachment)
        asyncio.run(bot.on_message(message))
        self.assertIn("処理中にエラーが発生しました", public_text(message))
        self.assertNotIn(str(self.root), public_text(message))
        self._assert_only_old_archive()

    def test_whisper_failure_cleans_temp(self):
        class Broken:
            def transcribe(self, audio_path, beam_size=5):
                raise RuntimeError("whisper failed")

        bot.whisper_model = Broken()
        message = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))
        asyncio.run(bot.on_message(message))
        self.assertIn("処理中にエラーが発生しました", public_text(message))
        self.assertNotIn("whisper failed", public_text(message))
        self._assert_only_old_archive()

    def test_discord_send_failure_cleans_temp(self):
        message = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))

        async def failing_send(content=None, **kwargs):
            if "file" in kwargs:
                raise bot.discord.DiscordException("upload failed")
            message.channel.sent.append((content, kwargs))

        message.channel.send = failing_send
        asyncio.run(bot.on_message(message))
        self.assertIn("処理中にエラーが発生しました", public_text(message))
        self.assertNotIn("校正済み", public_text(message))
        self._assert_only_old_archive()

    def test_same_filename_concurrent_jobs_do_not_collide(self):
        class SlowModel:
            def transcribe(self, audio_path, beam_size=5):
                data = Path(audio_path).read_bytes()
                return [Segment(data[-4:].decode("ascii"))], None

        bot.whisper_model = SlowModel()
        bot.correct_transcription = lambda text: (text, True)
        first = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))
        second = Message(42, Attachment("lecture.wav", wav_bytes(b"BBBB")))

        async def run():
            await asyncio.gather(bot.on_message(first), bot.on_message(second))

        asyncio.run(run())
        texts = [
            path.read_text(encoding="utf-8")
            for path in self.archives.rglob("*.txt")
            if path != self.old_file
        ]
        self.assertCountEqual(texts, ["AAAA", "BBBB"])
        wavs = [path for path in self.archives.rglob("*.wav") if path != self.old_file]
        self.assertEqual(len(wavs), 2)
        self.assertEqual(len({path.name for path in wavs}), 2)
        self.assertEqual(self.old_file.read_bytes(), b"keep-existing-archive")
        for job_dir in self.job_dirs:
            self.assertFalse(Path(job_dir).exists())

    def test_ollama_timeout_is_not_labeled_corrected(self):
        def boom(*args, **kwargs):
            raise requests.Timeout("timed out")

        bot.correct_transcription = self._correct
        bot.requests.post = boom
        message = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))
        asyncio.run(bot.on_message(message))
        text = public_text(message)
        self.assertIn("未校正", text)
        self.assertNotIn("校正済み", text)
        saved = next(path for path in self.archives.rglob("*.txt") if path != self.old_file)
        self.assertEqual(saved.read_text(encoding="utf-8"), "AAAA")

    def test_split_boundaries(self):
        empty = bot.split_discord_message("")
        self.assertEqual(empty, ["内容は空でした。"])
        self.assertTrue(empty[0].strip())

        exact = bot.split_discord_message("あ" * 2000)
        self.assertEqual(exact, ["あ" * 2000])
        self.assertEqual(bot.utf16_len(exact[0]), 2000)

        over = bot.split_discord_message("あ" * 2001)
        self.assertEqual([len(chunk) for chunk in over], [2000, 1])
        self.assertEqual("".join(over), "あ" * 2001)
        self.assertTrue(all(bot.utf16_len(chunk) <= 2000 for chunk in over))

        spaced = "A" + (" " * 4000) + "B"
        spaced_chunks = bot.split_discord_message(spaced)
        self.assertTrue(all(chunk.strip() for chunk in spaced_chunks))
        self.assertTrue(all(bot.utf16_len(chunk) <= 2000 for chunk in spaced_chunks))
        self.assertIn("A", "".join(spaced_chunks))
        self.assertIn("B", "".join(spaced_chunks))

        emoji = "😀" * 1500
        emoji_chunks = bot.split_discord_message(emoji)
        self.assertGreater(len(emoji_chunks), 1)
        self.assertEqual("".join(emoji_chunks), emoji)
        self.assertTrue(all(bot.utf16_len(chunk) <= 2000 for chunk in emoji_chunks))
        self.assertEqual(bot.split_discord_message("😀" * 1000), ["😀" * 1000])
        self.assertEqual(
            [bot.utf16_len(chunk) for chunk in bot.split_discord_message("😀" * 1001)],
            [2000, 2],
        )

    def test_invalid_allowed_user_ids(self):
        self.assertEqual(bot.load_allowed_user_ids(""), set())
        self.assertEqual(bot.load_allowed_user_ids("   "), set())
        self.assertEqual(bot.load_allowed_user_ids("123, 456"), {"123", "456"})
        with self.assertRaises(bot.ConfigurationError):
            bot.load_allowed_user_ids("123,abc")
        with self.assertRaises(bot.ConfigurationError):
            bot.load_allowed_user_ids("１２３")
        with self.assertRaises(bot.ConfigurationError):
            bot.load_allowed_user_ids("²")

        previous_token = bot.TOKEN
        previous_ids = os.environ.get("ALLOWED_USER_IDS")
        bot.TOKEN = "dummy"
        os.environ["ALLOWED_USER_IDS"] = "123,abc"

        def fail_if_called():
            raise AssertionError("whisper should not load")

        original_loader = bot.load_whisper_model
        bot.load_whisper_model = fail_if_called
        try:
            with self.assertRaises(SystemExit) as caught:
                bot.main()
            self.assertIn("設定エラー", str(caught.exception))
        finally:
            bot.load_whisper_model = original_loader
            bot.TOKEN = previous_token
            if previous_ids is None:
                os.environ.pop("ALLOWED_USER_IDS", None)
            else:
                os.environ["ALLOWED_USER_IDS"] = previous_ids

    def test_whisper_cancellation_blocks_the_next_caller(self):
        entered = []
        first_inside = threading.Event()
        release_first = threading.Event()
        second_waiting = threading.Event()
        inner = bot.whisper_model_lock

        class TrackingLock:
            def acquire(self, blocking=True, timeout=-1):
                if inner.locked():
                    second_waiting.set()
                return inner.acquire(blocking, timeout)

            def release(self):
                inner.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.release()

            def locked(self):
                return inner.locked()

        class Model:
            def transcribe(self, audio_path, beam_size=5):
                entered.append(("enter", audio_path))
                if audio_path == "first":
                    first_inside.set()
                    if not release_first.wait(3):
                        raise TimeoutError("first whisper was not released")
                entered.append(("exit", audio_path))
                return [Segment("x")], None

        bot.whisper_model_lock = TrackingLock()

        async def scenario():
            first = asyncio.create_task(asyncio.to_thread(bot.transcribe_audio, Model(), "first"))
            self.assertTrue(await asyncio.to_thread(first_inside.wait, 3))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertNotIn(("exit", "first"), entered)
            second = asyncio.create_task(asyncio.to_thread(bot.transcribe_audio, Model(), "second"))
            self.assertTrue(await asyncio.to_thread(second_waiting.wait, 3))
            self.assertNotIn(("enter", "second"), entered)
            release_first.set()
            await asyncio.wait_for(second, timeout=3)
            self.assertLess(entered.index(("exit", "first")), entered.index(("enter", "second")))

        try:
            asyncio.run(scenario())
        finally:
            release_first.set()

    def test_cancelled_job_cleans_temp_and_keeps_old_archive(self):
        started = threading.Event()
        release = threading.Event()

        class Model:
            def transcribe(self, audio_path, beam_size=5):
                started.set()
                release.wait(3)
                return [Segment("x")], None

        bot.whisper_model = Model()
        message = Message(42, Attachment("lecture.wav", wav_bytes(b"AAAA")))

        async def scenario():
            task = asyncio.create_task(bot.on_message(message))
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        try:
            asyncio.run(scenario())
        finally:
            release.set()
        self._assert_only_old_archive()

    def test_secret_scan(self):
        root = Path(__file__).resolve().parent
        patterns = [
            re.compile(r"[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}"),
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            re.compile(r"AKIA[0-9A-Z]{16}"),
            re.compile(r"sk-[A-Za-z0-9]{20,}"),
            re.compile(r"\b\d{17,20}\b"),
            re.compile(r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]+['\"]"),
        ]
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix in {".pyc"}:
                continue
            if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                self.assertIsNone(pattern.search(text), f"{pattern.pattern} in {path.name}")


if __name__ == "__main__":
    unittest.main()
