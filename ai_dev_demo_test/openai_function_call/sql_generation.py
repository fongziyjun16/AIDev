import json
import sqlite3

from utils import functions, chat_completion_request, pretty_print_conversation

conn = sqlite3.connect("data/chinook.db")
print("Open database successfully")

def get_table_names(conn):
    table_names = []
    tables = conn.execute("select name from sqlite_master where type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names

# print(get_table_names(conn))

def get_column_names(conn, table_name):
    column_names = []
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for column in columns:
        column_names.append(column[1])
    return column_names

# print(get_column_names(conn, "albums"))

def get_database_info(conn):
    table_dicts = []
    for table_name in get_table_names(conn):
        column_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": column_names})
    return table_dicts

# print(get_database_info(conn))

database_schema_dict = get_database_info(conn)

database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

# print(database_schema_string)

# 定义一个功能列表，其中包含一个功能字典，该字典定义了一个名为"ask_database"的功能，用于回答用户关于音乐的问题
functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about music. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

def ask_database(conn, query):
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results

def execute_function_call(message):
    if message["function_call"]["name"] == "ask_database":
        query = json.loads(message["function_call"]["arguments"])["query"]
        results = ask_database(conn, query)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results

# 创建一个空的消息列表
messages = [
    # 向消息列表中添加一个系统角色的消息，内容是 "Answer user questions by generating SQL queries against the Chinook Music Database."
    {"role": "system", "content": "Answer user questions by generating SQL queries against the Chinook Music Database."},
    # 向消息列表中添加一个用户角色的消息，内容是 "Hi, who are the top 5 artists by number of tracks?"
    {"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"}
]

# 使用 chat_completion_request 函数获取聊天响应
chat_response = chat_completion_request(messages, functions)

# 从聊天响应中获取助手的消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的消息添加到消息列表中
messages.append(assistant_message)

if assistant_message.get("function_call"):
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})

# pretty_print_conversation(messages)

# 向消息列表中添加一个用户的问题，内容是 "What is the name of the album with the most tracks?"
messages.append({"role": "user", "content": "What is the name of the album with the most tracks?"})

# 使用 chat_completion_request 函数获取聊天响应
chat_response = chat_completion_request(messages, functions)

# 从聊天响应中获取助手的消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的消息添加到消息列表中
messages.append(assistant_message)

# 如果助手的消息中有功能调用
if assistant_message.get("function_call"):
    # 使用 execute_function_call 函数执行功能调用，并获取结果
    results = execute_function_call(assistant_message)
    # 将功能的结果作为一个功能角色的消息添加到消息列表中
    messages.append({"role": "function", "content": results, "name": assistant_message["function_call"]["name"]})

# 使用 pretty_print_conversation 函数打印对话
pretty_print_conversation(messages)

