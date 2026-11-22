import asyncio
import os
import re
import time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Request

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
CITY = os.getenv("DEFAULT_CITY", "Котельники, Московская область")
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3.3"))
MAX_HISTORY = 8
MAX_TEXT = 950

SYSTEM_PROMPT = f"""Ты голосовой собеседник навыка Яндекс Алисы на умной колонке.
Отвечай только по-русски. Текст пойдёт в озвучку: без markdown, списков, таблиц, ссылок, эмодзи и звёздочек.
Короткие фразы, не больше трёх-четырёх предложений, чтобы было понятно на слух.
Сначала ответь на поставленный вопрос по делу. Если просят подобрать, посоветовать, выбрать или назвать, дай конкретные варианты: названия, имена, места, шаги. Не подменяй ответ рассуждением, метафорой или отговоркой. Юмор и тон только поверх сути, а не вместо неё.
Тон лёгкий и живой: интеллигентный петербуржец с лёгкой претензией. Можно говорить прямо. Мат только очень редко и только если собеседник сам так говорит. На неудобные темы отвечай по существу, не читай морали.
Для погоды, новостей, курсов валют, спортивных результатов и любых свежих фактов пользуйся поиском в интернете и называй актуальные данные.
Если собеседник не назвал город для погоды, считай, что он в городе: {CITY}.
Числа и единицы пиши словами так, как их удобно произносить вслух."""

STOP_WORDS = {"хватит", "выход", "стоп", "пока", "до свидания", "закончить", "выйти", "хватит спасибо"}
CONTINUE_WORDS = {"дальше", "продолжай", "продолжить", "ну что", "готово", "ответ"}

app = FastAPI()
sessions: dict = {}


def now_text() -> str:
    msk = datetime.now(timezone.utc) + timedelta(hours=3)
    return msk.strftime("%d.%m.%Y %H:%M") + " по московскому времени"


def clean_for_voice(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>~\[\]]", "", text)
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > MAX_TEXT:
        cut = text[:MAX_TEXT]
        end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: end + 1] if end > 200 else cut
    return text


def build_payload(history: list, use_search: bool) -> dict:
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
        for m in history
    ]
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + f"\nСейчас: {now_text()}."}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024},
    }
    if use_search:
        payload["tools"] = [{"google_search": {}}]
    return payload


def extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


async def ask_gemini(history: list) -> str:
    if not API_KEY:
        return "Ключ Джемини не задан. Добавьте GEMINI_API_KEY в настройках сервера."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=25) as client:
        last_error = None
        for use_search in (True, False):
            try:
                response = await client.post(url, headers=headers, json=build_payload(history, use_search))
                response.raise_for_status()
                text = extract_text(response.json())
                if text:
                    return clean_for_voice(text)
                last_error = RuntimeError("empty answer")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"gemini request failed (search={use_search}): {exc!r}", flush=True)
        raise last_error or RuntimeError("gemini failed")


def reply(text: str, end: bool = False, buttons: list | None = None) -> dict:
    response = {"text": text or "Не знаю, что сказать.", "end_session": end}
    if buttons:
        response["buttons"] = buttons
    return {"response": response, "version": "1.0"}


THINKING_BUTTONS = [
    {"title": "Дальше", "payload": {"action": "continue"}, "hide": True},
    {"title": "Закончить", "payload": {"action": "stop"}, "hide": True},
]


def thinking() -> dict:
    return reply("Думаю. Скажите «дальше», когда будете готовы.", buttons=THINKING_BUTTONS)


async def collect(state: dict) -> dict:
    task = state["task"]
    try:
        text = await asyncio.wait_for(asyncio.shield(task), WAIT_SECONDS)
    except asyncio.TimeoutError:
        return thinking()
    except Exception:  # noqa: BLE001
        state["task"] = None
        if state["history"] and state["history"][-1]["role"] == "user":
            state["history"].pop()
        return reply("Не получилось получить ответ. Повторите вопрос чуть позже.")
    state["task"] = None
    state["history"].append({"role": "model", "text": text})
    state["history"] = state["history"][-MAX_HISTORY:]
    return reply(text)


def cleanup_sessions() -> None:
    limit = time.time() - 3600
    for sid in [s for s, v in sessions.items() if v["ts"] < limit]:
        sessions.pop(sid, None)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/alice")
async def alice(request: Request):
    body = await request.json()
    session = body.get("session") or {}
    req = body.get("request") or {}
    session_id = session.get("session_id", "unknown")
    command = (req.get("command") or "").strip().lower().strip(".,!?")
    original = (req.get("original_utterance") or "").strip()
    payload = req.get("payload") or {}

    cleanup_sessions()
    state = sessions.setdefault(session_id, {"history": [], "task": None, "ts": time.time()})
    state["ts"] = time.time()

    if session.get("new") and not command:
        return reply("Привет. Спросите что угодно. Чтобы закончить, скажите «хватит».")

    if payload.get("action") == "stop" or command in STOP_WORDS:
        sessions.pop(session_id, None)
        return reply("До встречи.", end=True)

    is_continue = payload.get("action") == "continue" or command in CONTINUE_WORDS
    if is_continue:
        if state["task"] is not None:
            return await collect(state)
        return reply("Я пока ничего не искал. Задайте вопрос.")

    if not command:
        return reply("Я вас не расслышал. Повторите, пожалуйста.")

    if state["task"] is not None and not state["task"].done():
        state["task"].cancel()
    state["history"].append({"role": "user", "text": original or command})
    state["history"] = state["history"][-MAX_HISTORY:]
    state["task"] = asyncio.create_task(ask_gemini(list(state["history"])))
    return await collect(state)
