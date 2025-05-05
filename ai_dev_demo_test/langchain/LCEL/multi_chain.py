from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

planner = (
    ChatPromptTemplate.from_template("生成关于以下内容的论点：{input}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template("列出关于{base_response}的正面或有利的方面")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

arguments_against = (
    ChatPromptTemplate.from_template("列出关于{base_response}的反面或不利的方面")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages([
        ("ai", "{original_response}"),
        ("human", "正面观点:\n{results_1}\n\n反面观点:\n{results_2}"),
        ("system", "给出批评后生成最终回应"),
    ])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    } | final_responder
)

# print(chain.invoke({"input": "小学生学习压力大"}))

for s in chain.strem({"input": "小学生学习压力大"}):
    print(s, end="", flush=True)




