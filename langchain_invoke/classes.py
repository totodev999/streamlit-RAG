from typing import Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
import uuid


class CustomCallbackManager(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        id = data["run_id"]
        # st: streamlit = data["st"]
        rest = {key: value for key, value in data.items() if key != "run_id"}
        # st.write(f"Received custom event: {name} with data: {rest} {id}")
        print(f"Received custom event: {name} with data: {rest} {id}")
