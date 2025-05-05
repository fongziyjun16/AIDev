from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings))
print(len(embeddings[0]))
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(len(embedded_query))