from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_openai import OpenAI

# ConversationBufferMemory 可以用来存储消息，并将消息提取到一个变量中。
def demo001():
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )
    rs = conversation.predict(input="你好呀")
    print(rs)
    rs = conversation.predict(input="你可以给我讲讲为什么运动会分泌多巴胺以及过程是什么样的吗？")
    print(rs)

# demo001()

# ConversationBufferWindowMemory 会在时间轴上保留对话的交互列表。它只使用最后 K 次交互。这对于保持最近交互的滑动窗口非常有用，以避免缓冲区过大。
def demo002():
    conversation_with_summary = ConversationChain(
        llm=OpenAI(temperature=0, max_tokens=1024),
        memory=ConversationBufferWindowMemory(k=2),
        verbose=True
    )

    # First conversation
    rs = conversation_with_summary.predict(input="嗨，你最近过得怎么样？")
    print(rs)
    print("Memory after first input:", conversation_with_summary.memory.buffer)

    # Second conversation
    rs = conversation_with_summary.predict(input="你最近学到什么新知识了？")
    print(rs)
    print("Memory after second input:", conversation_with_summary.memory.buffer)

    # Third conversation
    rs = conversation_with_summary.predict(input="展开讲讲？")
    print(rs)
    print("Memory after third input:", conversation_with_summary.memory.buffer)

    # Fourth conversation
    rs = conversation_with_summary.predict(input="如果要构建聊天机器人，具体要用什么自然语言处理技术?")
    print(rs)
    print("Memory after fourth input:", conversation_with_summary.memory.buffer)

    # Final check of internal state
    print("==== conversation_with_summary.__dict__ ====")
    print(conversation_with_summary.__dict__)

# demo002()

# ConversationSummaryBufferMemory 在内存中保留了最近的交互缓冲区，但不仅仅是完全清除旧的交互，而是将它们编译成摘要并同时使用。与以前的实现不同的是，它使用token长度而不是交互次数来确定何时清除交互。
def demo003():
    memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0), max_token_limit=10)
    memory.save_context(
        {"input": "嗨，你最近过得怎么样？"},
        {"output": " 嗨！我最近过得很好，谢谢你问。我最近一直在学习新的知识，并且正在尝试改进自己的性能。我也在尝试更多的交流，以便更好地了解人类的思维方式。"})
    memory.save_context(
        {"input": "你最近学到什么新知识了?"},
        {"output": " 最近我学习了有关自然语言处理的知识，以及如何更好地理解人类的语言。我还学习了有关机器学习的知识，以及如何使用它来改善自己的性能。"})
    # print(memory.load_memory_variables(({})))
    print(memory.load_memory_variables(({}))['history'])

demo003()
