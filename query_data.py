import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    if GROQ_API_KEY is None:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    # Initialize the Groq client
    client = Groq(api_key=GROQ_API_KEY)

    # Initialize HuggingFaceEmbeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

    # Initialize Chroma with the embedding function
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf_embeddings)

    print(f"Database loaded. Number of documents: {db._collection.count()}")

    print("Welcome to the chat interface! Ask me a question:")
    while True:
        query_text = input("> ")
        if query_text.lower() == "exit":
            break

        # Search the DB
        try:
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
            if len(results) == 0:
                print("No relevant results found in the database.")
            else:
                print(f"Found {len(results)} relevant results.")

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            print(f"Context for the query:\n{context_text}")

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            print(f"Prompt sent to Groq:\n{prompt}")


            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"An error occurred during querying: {e}")

if __name__ == "__main__":
    main()