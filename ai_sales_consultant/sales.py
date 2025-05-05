from typing import List

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

with open("real_estate_sales_data.txt", "r", encoding="utf-8") as f:
    real_estate_sales = f.read()
text_splitter = CharacterTextSplitter(
    separator=r"\n\d+\.\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=True,
)
docs = text_splitter.create_documents([real_estate_sales])

db_folder_path = "real_estates_sale"
db = FAISS.from_documents(docs, OpenAIEmbeddings())

# answer_list = db.similarity_search("小区吵不吵")
# for ans in answer_list:
#     print(ans.page_content + "\n")

# save data to folder
# db.save_local(db_folder_path)
# after saving data to folder above, no need to open txt file and load data again
# db = FAISS.load_local(db_folder_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def sales(query: str, score_threshold: float=0.8) -> List[str]:
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold})
    docs = retriever.get_relevant_documents(query)
    ans_list = [doc.page_content.split("[销售回答] ")[-1] for doc in docs]

    return ans_list

# print(sales("我想离医院近点", 0.4))

llm = ChatOpenAI(model="gpt-4", temperature=0.5)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.64}
    ),
    return_source_documents=True,
    chain_type_kwargs={"verbose": True}
)

# print(qa_chain({"query": "我想买别墅，你们有么"}))
print(qa_chain({"query": "小区吵不吵"}))
# print(qa_chain({"query": "你们小区有200万的房子吗？"}))




