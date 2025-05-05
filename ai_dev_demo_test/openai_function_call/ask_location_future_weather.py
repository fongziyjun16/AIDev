from utils import functions, chat_completion_request, pretty_print_conversation

# 定义一个空列表messages，用于存储聊天的内容
messages = [{
    "role": "system",  # 消息的角色是"system"
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    # 消息的内容
}, {
    "role": "user",  # 消息的角色是"user"
    "content": "what is the weather going to be like in Guangzhou, China over the next x days"  # 用户询问今天的天气情况
}]

# 使用append方法向messages列表添加一条系统角色的消息

# 向messages列表添加一条用户角色的消息

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取助手的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# pretty_print_conversation(messages)

# 向messages列表添加一条用户角色的消息，用户指定接下来的天数为5天
messages.append({
    "role": "user",  # 消息的角色是"user"
    "content": "5 days"
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

