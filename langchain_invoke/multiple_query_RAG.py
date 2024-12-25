from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document


import streamlit as st
from pydantic import BaseModel, Field

from .classes import CustomCallbackManager


class QueryGenerateModelOutput(BaseModel):
    queries: list[str] = Field(..., title="検索クエリのリスト")


# To avoid the below error, separate the function into some chains.
# Thread 'ThreadPoolExecutor-1_0': missing ScriptRunContext! This warning can be ignored when running in bare mode.
# The below discussion cloud be helpful?
# https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893/7


def query_generate_model(text: str, run_id: str):
    prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。
                                                           
質問: {question}
""")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def get_queries(x: QueryGenerateModelOutput):
        return x.queries

    chain = (
        prompt | model.with_structured_output(QueryGenerateModelOutput) | get_queries
    )

    queries = chain.invoke(text, run_id=run_id)

    st.json(queries)

    return queries


def sample_retriever(input: str):
    # dispatch_custom_event(name="retriever", data=input)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name="japanese_companies",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    return db.as_retriever().invoke(input)


def sort_documents(retriever_output: list[list[Document]], k=60) -> list[Document]:
    content_score_mapping = {}

    # ドキュメントごとにスコアを計算して保存
    for docs in retriever_output:
        for rank, doc in enumerate(docs):
            if doc.page_content not in content_score_mapping:
                content_score_mapping[doc.page_content] = {"document": doc, "score": 0}
            # スコアを更新
            content_score_mapping[doc.page_content]["score"] += 1 / (rank + k)
    st.write("content_score_mapping")
    st.json(content_score_mapping)
    # スコアでランキング
    ranked = sorted(
        content_score_mapping.values(), key=lambda x: x["score"], reverse=True
    )

    # 上位3つのドキュメントを返す
    return [item["document"] for item in ranked][:3]


def embedding_and_retriever(text: list[str], run_id: str):
    config = RunnableConfig({"callbacks": [CustomCallbackManager()], "run_id": run_id})

    def list_query_retriever(queries: dict[str, str]):
        print(f"{queries} is passed to list_query_retriever")
        return list(map(sample_retriever, queries["queries"]))

    chain = RunnableLambda(list_query_retriever) | sort_documents

    documents = chain.invoke({"queries": text, "run_id": run_id}, config=config)

    st.write("documents")
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


def invoke_multiple_query_RAG(text: str, run_id: str):
    queries = query_generate_model(text=text, run_id=run_id)
    documents = embedding_and_retriever(text=queries, run_id=run_id)
    prompt = create_prompt(question=text, context=documents, run_id=run_id)
    get_answer(prompt=prompt, run_id=run_id)
    return prompt, documents
