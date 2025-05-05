from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def text_split():
    with open("state_of_the_union.txt", encoding="utf-8") as f:
        state_of_the_union = f.read()
    # print(state_of_the_union)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )

    docs = text_splitter.create_documents([state_of_the_union])
    # print(docs[0])
    # print(docs[1])

    metadatas = [{"document": 1}, {"document": 2}]
    documents = text_splitter.create_documents(
        [state_of_the_union, state_of_the_union],
        metadatas=metadatas
    )
    print(documents[0])

# text_split()

def code_split():
    # print([e.value for e in Language])

    html_text = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>🦜️🔗 LangChain</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                h1 {
                    color: darkblue;
                }
            </style>
        </head>
        <body>
            <div>
                <h1>🦜️🔗 LangChain</h1>
                <p>⚡ Building applications with LLMs through composability ⚡</p>
            </div>
            <div>
                As an open source project in a rapidly developing field, we are extremely open to contributions.
            </div>
        </body>
    </html>
    """

    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=60, chunk_overlap=0
    )
    html_docs = html_splitter.create_documents([html_text])
    print(len(html_docs))
    print(html_docs)

code_split()
