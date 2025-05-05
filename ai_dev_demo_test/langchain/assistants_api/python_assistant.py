import os
import time

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个名为 "Python Master" 的助手，它能根据需求生成可以运行的 Python 代码
assistant_python = openai.beta.assistants.create(
    name="Python Master",
    instructions="You are a Python Expert. Generate runnable Python code according to messages.",
    tools=[{"type": "code_interpreter"}],  # 使用工具：代码解释器
    model="gpt-4o",  # 使用模型： GPT-4
)

thread_python = openai.beta.threads.create()

message = openai.beta.threads.messages.create(
    thread_id=thread_python.id,
    role="user",
    content="编写快速排序算法代码",
)

run = openai.beta.threads.runs.create(
    thread_id=thread_python.id,
    assistant_id=assistant_python.id,
)

while run.status == "queued" or run.status == "in_progress":
    run = openai.beta.threads.runs.retrieve(
        thread_id=thread_python.id,
        run_id=run.id
    )
    time.sleep(1)
print("run completed with status: " + run.status)  # 打印执行流的完成状态

# 如果执行流状态为 "completed"（已完成），则获取并打印所有消息
if run.status == "completed":
    messages = openai.beta.threads.messages.list(thread_id=thread_python.id)

    print("\nMessages:\n")
    for message in messages:
        assert message.content[0].type == "text"
        print(f"Role: {message.role.capitalize()}")  # 角色名称首字母大写
        print("Message:")
        print(message.content[0].text.value + "\n")  # 每条消息后添加空行以增加可读性
