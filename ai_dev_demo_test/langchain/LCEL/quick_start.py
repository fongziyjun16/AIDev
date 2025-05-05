from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def quick_start():
    model = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template("讲个关于 {topic} 的笑话吧")

    output_parse = StrOutputParser()

    chain = prompt | model | output_parse

    print(chain.invoke({"topic": "程序员"}))

def rag_demo():
    model = ChatOpenAI(model="gpt-4o-mini")

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )

    retriever = vectorstore.as_retriever()

    template = """根据以下上下文回答问题:
    {context}

    问题: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    output_parse = StrOutputParser()

    # 设置一个并行运行器，用于同时处理上下文检索和问题传递
    # 使用RunnableParallel来准备预期的输入，通过使用检索到的文档条目以及原始用户问题，
    # 利用文档搜索器 retriever 进行文档搜索，并使用 RunnablePassthrough 来传递用户的问题。
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | model | output_parse

    print(chain.invoke("harrison 在哪里工作？"))

rag_demo()

