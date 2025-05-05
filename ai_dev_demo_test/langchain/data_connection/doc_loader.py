from langchain_community.document_loaders import TextLoader, ArxivLoader, UnstructuredURLLoader


def load_txt():
    docs = TextLoader("state_of_the_union.txt", encoding="utf-8").load()
    print(docs)
    print(docs[0].page_content[:100])

# load_txt()

def load_arxiv():
    query = "2005.14165"
    docs = ArxivLoader(query=query, load_max_docs=5).load()
    print(len(docs))
    print(docs[0].metadata)

# load_arxiv()

def load_web_page():
    urls = ["https://react-lm.github.io/"]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    print(data[0].metadata)
    print(data[0].page_content)

    loader = UnstructuredURLLoader(urls=urls, mode="elements")
    new_data = loader.load()
    print(new_data[0].page_content)
    print(len(new_data))
    print(new_data[1].page_content)

# load_web_page()

