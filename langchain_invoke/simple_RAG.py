from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

import streamlit as st

from .classes import CustomCallbackManager

# To avoid the below error, separate the function into some chains.
# Thread 'ThreadPoolExecutor-1_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# The below discussion cloud be helpful?
# https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893/7


def sample_retriever(input: dict[str, str]):
    # dispatch_custom_event(name="retriever", data=input)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name="japanese_companies",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    return db.as_retriever().invoke(input["question"])


def embedding_and_retriever(text: str, run_id: str):
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})

    chain = RunnableLambda(sample_retriever)

    documents = chain.invoke({"question": text, "run_id": run_id}, config=config)

    st.json(documents)
    return documents


def create_prompt(
    question: str, context: list[Document], run_id: str
) -> ChatPromptTemplate:
    # dispatch_custom_event(name="create_prompt", data=input)
    prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})
    chain = {
        "question": lambda x: x["question"],
        "context": lambda x: x["context"],
    } | prompt
    prompt_result = chain.invoke(
        {"question": question, "context": context}, config=config
    )

    st.json(prompt_result)
    return prompt_result


def get_answer(prompt: ChatPromptTemplate, run_id: str):
    # dispatch_custom_event(name="get_answer", data=input)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})
    answer = model.invoke(prompt, config=config)

    st.json(answer)
    return answer


def invoke_simple_RAG(text: str, run_id: str):
    documents = embedding_and_retriever(text=text, run_id=run_id)
    prompt = create_prompt(question=text, context=documents, run_id=run_id)
    get_answer(prompt=prompt, run_id=run_id)
    return prompt, documents
