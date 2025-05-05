import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个名为 "Mark Johnson" 的助手，它是一个个人数学辅导老师。这个助手能够编写并运行代码来解答数学问题。
assistant = openai.beta.assistants.create(
    name="Mark Johnson",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o"
)

thread = openai.beta.threads.create()

message = openai.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

run = openai.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    # 以 Jane Doe 称呼用户，并且用户拥有高级账户
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)

# # 打印执行流的完成状态
# print("Run completed with status: " + run.status)
#
# # 如果执行流状态为 "completed"（已完成），则获取并打印所有消息
# if run.status == "completed":
#     messages = openai.beta.threads.messages.list(thread_id=thread.id)
#
#     print("\nMessages:\n")
#     for message in messages:
#         assert message.content[0].type == "text"
#         print(f"Role: {message.role.capitalize()}")  # 角色名称首字母大写
#         print("Message:")
#         print(message.content[0].text.value + "\n")  # 每条消息后添加空行以增加可读性

# Example of continuing the conversation by adding another user question

new_question = "Can you solve the equation `5x - 3 = 22` for me?"

# Send the new message in the same thread
new_message = openai.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",  # Since the user is asking the question
    content=new_question
)

# Optionally, you can also trigger the assistant's response to the new question by running the assistant with the same instructions
run = openai.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)

# Print status and messages if the run is completed
print("Run completed with status: " + run.status)

if run.status == "completed":
    messages = openai.beta.threads.messages.list(thread_id=thread.id)

    print("\nMessages:\n")
    for message in messages:
        assert message.content[0].type == "text"
        print(f"Role: {message.role.capitalize()}")  # Capitalizing the role
        print("Message:")
        print(message.content[0].text.value + "\n")

openai.beta.assistants.delete(assistant.id)