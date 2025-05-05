import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def test1():
    data = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Say this is a test",
        max_tokens=8,
        temperature=0
    )
    print(data)
    res = data.choices[0].text
    print(res)

# test1()

def test2():
    data = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=8,
        temperature=0,
        messages=[
            {"role": "user", "content": "Say this is a test"}
        ]
    )
    print(data)
    res = data.choices[0].message.content
    print(res)

# test2()

def test3():
    data = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="讲 8 个给程序员听的笑话",
        max_tokens=1024,
        temperature=0.64
    )
    print(data)
    res = data.choices[0].text
    print(res)

# test3()

def test4():
    data = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.64,
        messages=[
            {"role": "user", "content": "讲 8 个给程序员听的笑话"}
        ]
    )
    print(data)
    res = data.choices[0].message.content
    print(res)

# test4()

def test5():
    data = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="生成可执行的快速排序 Python 代码",
        max_tokens=1024,
        temperature=0
    )
    print(data)
    res = data.choices[0].text
    print(res)

# test5()

def test6():
    data = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "user", "content": "生成可执行的快速排序 Python 代码，只输出代码块内容，函数名 quick_sort"}
        ]
    )
    print(data)
    res = data.choices[0].message.content
    print(res)

# test6()
