from utils import functions, chat_completion_request, pretty_print_conversation

# 在这个代码单元中，我们强制GPT 模型使用get_n_day_weather_forecast函数
messages = [{
    "role": "system",  # 角色为系统
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
}, {
    "role": "user",  # 角色为用户
    "content": "Give me a weather report for San Diego, USA."
}]  # 创建一个空的消息列表

# 添加系统角色的消息

# 添加用户角色的消息

# 使用定义的chat_completion_request函数发起一个请求，传入messages、functions以及特定的function_call作为参数
chat_response = chat_completion_request(
    messages, functions=functions, function_call={"name": "get_n_day_weather_forecast"}

    # 如果我们不强制GPT 模型使用 get_n_day_weather_forecast，它可能不会使用
    # messages, functions=functions
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)