import streamlit as st
from langchain_invoke.simple_RAG import invoke_simple_RAG
from langchain_invoke.multiple_query_RAG import invoke_multiple_query_RAG
from utils.set_env import set_envs
import uuid

invoke_settings = {
    "simple_RAG": invoke_simple_RAG,
    "multiple_query_RAG": invoke_multiple_query_RAG,
}


def main():
    st.title("Hello world")

    with st.form(key="my_form"):
        text = st.text_area("質問を入力してください")
        selected = st.selectbox(
            "モデルを選択してください", list(invoke_settings.keys())
        )
        submited = st.form_submit_button("送信")

    if submited:
        run_id = uuid.uuid4()
        set_envs()
        invoke_settings[selected](text=text, run_id=run_id)


if __name__ == "__main__":
    main()
