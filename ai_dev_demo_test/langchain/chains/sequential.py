from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# basic
def basic_demo():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=512)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="给主题{topic}游乐场取8个好名字，并给出完整公司名称"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    chain.verbose = True
    print(chain.invoke({"topic": "狂野飙车"}))

# basic_demo()

# sequential chain
def seq_chain_demo():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=500)

    template = """
        你是一位记者。根据新闻标题，你的任务是为该标题写一个新闻概要。
        标题：{title}
        记者：以下是对上述戏剧的简介：
    """
    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    news_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """
        你是新闻报道的新闻评论员。根据新闻概要，你的工作是为该概要写一篇评论。
        新闻概要：
        {news}
        以下是来自新闻报道的新闻评论员对上述新闻概要的评论：
    """
    prompt_template = PromptTemplate(input_variables=["news"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(
        chains=[news_chain, review_chain],
        verbose=True
    )

    # review = overall_chain.invoke({"title": "一男子制造出小型化可控核聚变设备"})
    review = overall_chain.invoke("一男子制造出小型化可控核聚变设备")
    print(review)

# seq_chain_demo()

def seq_chain_demo_with_multiple_inputs():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=500)

    template = """你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。

    标题：{title}
    时代：{era}
    剧作家：以下是对上述戏剧的简介："""

    prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
    # output_key
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis", verbose=True)

    # 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。

    template = """你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。
    
    剧情简介：
    {synopsis}
    
    来自《纽约时报》戏剧评论家对上述剧目的评价："""

    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", verbose=True)

    m_overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True)

    result = m_overall_chain.invoke({"title": "三体人不是无法战胜的", "era": "二十一世纪的新中国"})
    print(result)

seq_chain_demo_with_multiple_inputs()
