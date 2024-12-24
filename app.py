import streamlit as st
from langchain_invoke.langchain_invoke import (
    set_envs,
    embedding_and_retriever,
    create_prompt,
    get_answer,
)
import uuid


def main():
    st.title("Hello world")

    with st.form(key="my_form"):
        text = st.text_area("質問を入力してください")
        submited = st.form_submit_button("送信")

    if submited:
        run_id = uuid.uuid4()
        set_envs()
        documents = embedding_and_retriever(text=text, run_id=run_id)
        st.json(documents)

        prompt_template = create_prompt(question=text, context=documents, run_id=run_id)
        prompt = prompt_template.messages[0]
        st.json(prompt)

        answer = get_answer(prompt=prompt_template, run_id=run_id)
        st.json(answer)


if __name__ == "__main__":
    main()
