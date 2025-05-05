from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    max_tokens=1024
)

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

prompt = prompt_template.format(
    adjective="funny",
    content="chickens"
)

# result = llm.invoke(prompt)
# print(f"result: {result}")

prompt_template = PromptTemplate.from_template(
    "讲{num}个给程序员听的笑话"
)

prompt = prompt_template.format(num=2)

# print(f"result: {llm.invoke(prompt)}")

jinja2_template = "Tell me a {{ adjective }} joke about {{ content }}"
prompt_template = PromptTemplate.from_template(jinja2_template, template_format="jinja2")
prompt = prompt_template.format(
    adjective="funny",
    content="Marvel"
)
# print(llm.invoke(prompt))

sort_alg_prompt_template = PromptTemplate.from_template("生成可执行的快速排序 {programming_language} 代码")
# print(llm.invoke(sort_alg_prompt_template.format(programming_language="rust")))



