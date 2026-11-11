"""LLM-агент «Менеджер справ»: локальна LLM (Ollama) + інструменти + Streamlit."""

import json
from datetime import date, timedelta
from pathlib import Path

import ollama
import streamlit as st

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"
DB_PATH = Path(__file__).with_name("tasks.json")

SYSTEM_PROMPT = (
    "Ти — дружній асистент, що веде список справ користувача. "
    "Для будь-яких дій зі справами ЗАВЖДИ використовуй інструменти, нічого не вигадуй. "
    "Календар на найближчий тиждень (перший день — сьогодні): {week}. "
    "Дати передавай у форматі YYYY-MM-DD (для «завтра», «у п'ятницю» бери дату з календаря; "
    "якщо рік не вказано — бери найближчу майбутню дату). "
    "Перш ніж виконати чи видалити справу, дізнайся її id через list_tasks. "
    "Відповідай коротко, українською мовою."
)


# Сховище (JSON-файл замість бази даних)

def load_tasks() -> list[dict]:
    return json.loads(DB_PATH.read_text(encoding="utf-8")) if DB_PATH.exists() else []


def save_tasks(tasks: list[dict]) -> None:
    DB_PATH.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def find_task(tasks: list[dict], task_id) -> dict:
    for task in tasks:
        if task["id"] == int(task_id):
            return task
    raise ValueError(f"task with id={task_id} not found")


# Інструменти агента

def add_task(title: str, due: str) -> str:
    """Add a new task to the to-do list.

    Args:
      title: Short description of the task
      due: Deadline in YYYY-MM-DD format, or an empty string if there is no deadline
    """
    if due:
        date.fromisoformat(due)  # некоректна дата -> помилка повернеться моделі
    tasks = load_tasks()
    task = {"id": max((t["id"] for t in tasks), default=0) + 1,
            "title": title, "due": due, "done": False}
    tasks.append(task)
    save_tasks(tasks)
    return f"Added: {json.dumps(task, ensure_ascii=False)}"


def list_tasks(include_done: bool) -> str:
    """Return tasks from the to-do list as JSON.

    Args:
      include_done: If true, also return tasks that are already completed
    """
    tasks = [t for t in load_tasks() if include_done or not t["done"]]
    return json.dumps(tasks, ensure_ascii=False) if tasks else "The list is empty."


def complete_task(task_id: int) -> str:
    """Mark a task as completed.

    Args:
      task_id: Id of the task
    """
    tasks = load_tasks()
    find_task(tasks, task_id)["done"] = True
    save_tasks(tasks)
    return f"Task {task_id} marked as done."


def delete_task(task_id: int) -> str:
    """Delete a task from the list.

    Args:
      task_id: Id of the task
    """
    tasks = load_tasks()
    tasks.remove(find_task(tasks, task_id))
    save_tasks(tasks)
    return f"Task {task_id} deleted."


TOOLS = {f.__name__: f for f in [add_task, list_tasks, complete_task, delete_task]}


# Цикл агента

def run_agent(messages: list[dict], model: str, host: str, max_steps: int = 10) -> None:
    """Викликає LLM, виконує запитані нею інструменти й повторює, доки не буде фінальної відповіді.
    Усі нові повідомлення (відповіді моделі та результати інструментів) додаються в messages."""
    client = ollama.Client(host=host)
    for _ in range(max_steps):
        response = client.chat(model=model, messages=messages,
                               tools=list(TOOLS.values()), think=False)
        messages.append(response.message.model_dump(exclude_none=True))
        if not response.message.tool_calls:
            return
        for call in response.message.tool_calls:
            try:
                result = str(TOOLS[call.function.name](**call.function.arguments))
            except Exception as e:  # помилку повертаємо моделі, щоб вона могла виправитися
                result = f"Error: {e}"
            messages.append({"role": "tool", "tool_name": call.function.name, "content": result})


# Інтерфейс Streamlit

st.set_page_config(page_title="Агент-менеджер справ", page_icon="✅")
st.title("✅ Агент-менеджер справ")
st.write(
    "Пишіть звичайною мовою: *«додай купити молоко на завтра»*, *«що в мене сьогодні?»*, "
    "*«я вже купив молоко»*. Локальна LLM (через [Ollama](https://ollama.com/)) сама вирішує, "
    "які інструменти викликати: `add_task`, `list_tasks`, `complete_task`, `delete_task`."
)

with st.sidebar:
    st.header("Налаштування моделі")
    host = st.text_input("Адреса Ollama сервера", DEFAULT_HOST)
    model = st.text_input("Модель (з підтримкою tools)", DEFAULT_MODEL)
    if st.button("Очистити чат"):
        st.session_state.pop("messages", None)
    st.header("Поточні справи")
    tasks_placeholder = st.empty()

if "messages" not in st.session_state:
    week = ", ".join((date.today() + timedelta(days=i)).strftime("%A %Y-%m-%d") for i in range(7))
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT.format(week=week)}]
messages = st.session_state.messages


def render(msg: dict) -> None:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            for call in msg.get("tool_calls", []):
                args = ", ".join(f"{k}={v!r}" for k, v in call["function"]["arguments"].items())
                st.caption(f"🔧 `{call['function']['name']}({args})`")
            if msg.get("content"):
                st.write(msg["content"])
    elif msg["role"] == "tool":
        st.caption(f"↳ {msg['content']}")


for msg in messages:
    render(msg)

if prompt := st.chat_input("Що потрібно зробити?"):
    messages.append({"role": "user", "content": prompt})
    render(messages[-1])
    start = len(messages)
    with st.spinner("Агент думає..."):
        try:
            run_agent(messages, model, host)
        except Exception as e:
            st.error(f"Помилка звернення до моделі: {e}")
    for msg in messages[start:]:
        render(msg)

tasks_placeholder.dataframe(load_tasks(), hide_index=True, use_container_width=True)
