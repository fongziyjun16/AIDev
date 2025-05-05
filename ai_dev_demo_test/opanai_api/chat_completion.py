import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def chat(msgs):
    data = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=msgs
    )
    new_message = data.choices[0].message
    return {
        "role": new_message.role,
        "content": new_message.content
    }

def test1():
    messages = [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]

    # begin
    new_message = chat(messages)
    messages.append(new_message)
    print(new_message)

    # new chat
    new_chat = {
        "role": "user",
        "content": "1.讲一个程序员才听得懂的冷笑话；2.今天是几号？3.明天星期几？"
    }
    messages.append(new_chat)
    new_message = chat(messages)
    messages.append(new_message)
    print(new_message)

    print()
    print(messages)

# test1()

def test2():
    messages = [
        {"role": "system", "content": "你是一个乐于助人的体育界专家。"},
        {"role": "user", "content": "2008年奥运会是在哪里举行的？"},
    ]

    #begin
    new_message = chat(messages)
    messages.append(new_message)
    print(new_message)

    # new chat
    messages.append({"role": "user", "content": "1.金牌最多的是哪个国家？2.奖牌最多的是哪个国家？"})
    new_message = chat(messages)
    messages.append(new_message)
    print(new_message)

    print()
    print(messages)

test2()
