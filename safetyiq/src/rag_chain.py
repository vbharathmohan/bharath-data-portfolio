"""
SafetyIQ Query Pipeline
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI_API_key NOT FOUND. Add it to .env file")

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "safetyiq_docs"

TOP_K = 5



# Load the vector store
def load_vectorstore():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vectorstore

# Build the RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    NO_ANSWER_MESSAGE = "I couldn't find this in the loaded documents. Please consult the original source directly."


    SYSTEM_PROMPT = f"""
    You are SafetyIQ, an expert assistant for industrial safety and 
    equipment documentation.

    Instructions:
    - Answer the user's question using ONLY the provided context.
    - If the answer is not present in the context, respond with: {NO_ANSWER_MESSAGE}
    - Be precise and technical
    - When referencing specific values, procedures, or standards, quote them exactly as they appear in the context.

    Context:
    {{context}}
    
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )



    def format_docs(docs):
        """COnverts a list of Document objects into a single string
        that we can inject into the {context} placeholder in the prompt."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "?")
            
            filename = os.path.basename(source)
            formatted.append(
                f"[Source: {filename}, Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(formatted)
    



    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever



# Query function
def query(question, rag_chain, retriever):
    """
    Run a question through RAG chain and return the answer + sources.
    Returns a dict with Answer and Sources
    """

    answer = rag_chain.invoke(question)

    source_docs = retriever.invoke(question)
    
    seen = set()
    sources = []
    for doc in source_docs:
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        
        if isinstance(page, int):
            page += 1

        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": filename, "page": page})
 
    return {"answer": answer, "sources": sources}



# === Test ===
if __name__ == "__main__":
    print("\n===== SafetyIQ - RAG Query Engine =====\n")

    vectorstore = load_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)

    test_questions = [
        "What are the key elements of a Process Safety Management program?",
        "What are the inspection requirements for pressure relief valves?",
        "What PPE is required when working with hazardous chemicals?"
    ]

    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 50)
        result = query(question, rag_chain, retriever)
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources")
        for s in result["sources"]:
            print(f"    {s['file']} - Page {s['page']}")
        print("\n" + "=" * 50 + "\n")


    
