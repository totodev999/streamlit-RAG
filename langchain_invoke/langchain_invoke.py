from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

from .classes import CustomCallbackManager


def set_envs():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_key


def sample_retriever(input: dict[str, str]):
    # dispatch_custom_event(name="retriever", data=input)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name="japanese_companies",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    return db.as_retriever().invoke(input["question"])


# To avoid the below error, separate the function into two chains.
# Thread 'ThreadPoolExecutor-1_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# The below discussion cloud be helpful?
# https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893/7
def embedding_and_retriever(text: str, run_id: str):
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})

    chain = RunnableLambda(sample_retriever)

    return chain.invoke({"question": text, "run_id": run_id}, config=config)


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
    return chain.invoke({"question": question, "context": context}, config=config)


def get_answer(prompt: ChatPromptTemplate, run_id: str):
    # dispatch_custom_event(name="get_answer", data=input)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})
    return model.invoke(prompt, config=config)
