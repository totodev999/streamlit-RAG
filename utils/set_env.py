import os
from dotenv import load_dotenv


def set_envs():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_key
