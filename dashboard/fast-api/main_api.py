import pathlib

from pydantic import BaseModel, Field
from typing import Union, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import logging

from StructuredRag.algorithms.inquirer import StructRAGInquirer

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    # filename=f"log/{SESSION_NAME}.log",
    filemode="a",
)

MISTRAL_MODEL_PATH = pathlib.Path(
    r"C:\Users\335257\.cache\huggingface\hub\models--TheBloke--CapybaraHermes-2.5-Mistral-7B-GGUF\snapshots\234067be357852d0c75bf1d04d2c720d15eab3e2\capybarahermes-2.5-mistral-7b.Q5_0.gguf"
)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
EXPERIMENT_PATH = "v0/2024-06-08"


inquirer = StructRAGInquirer(
    path_to_experiment=str(PROJECT_ROOT / "results" / EXPERIMENT_PATH),
    llm_type="llamacpp",
    # model_path=str(PROJECT_ROOT / "capybarahermes-2.5-mistral-7b.Q5_K_M.gguf"),
    model_path=str(MISTRAL_MODEL_PATH),
    llm_max_tokens=1600,
    n_gpu_layers=0,  # All layers CPU
    use_anchor_document=False,
    n_threads=4,
)

app = FastAPI(
    title="Structured RAG API",
    description="Experimental search using Structured RAG for the sample data",
    summary="""Experimental search of MA Notes and Data publications.
        Using retrieval augmented generation (RAG).""",
    version="0.0.2",
    contact={
        "name": "Lukas Alemu",
        "email": "lukas.alemu@bankofengland.co.uk",
    },
)


@app.get("/", tags=["Principle Endpoints"])
async def about():
    """Access the API documentation in json format.

    Returns:
        Redirect to /openapi.json
    """
    response = RedirectResponse(url="/openapi.json")
    return response


@app.get("/get_doc_list", tags=["Principle Endpoints"])
async def get_doc_list():
    """Get the list of available documents.

    Returns:
        List of available documents.
    """
    return inquirer.get_document_name_list()


@app.get("/search", tags=["Principle Endpoints"])
async def search(
    q: str,
    content_type: Union[str, None] = "latest",
    selected_document: Optional[str] = None,
    debug: bool = True,
):
    """Search api endpoint

    Args:
        q (str): query string.
        content_type (Union[str, None], optional): Flag for optional date preferencing. Not implemented. Defaults to "latest".
        selected_document (Optional[str], optional): Which document to use as an anchor. Defaults to None.
        debug (bool, optional): Not implemented. Defaults to True.

    Returns:
        Dict: Dictionary of results, including question, answer, references, and selected document.
    """
    print("Selected doc is", selected_document)
    response = inquirer.run_inquirer(
        query=q,
        source_document_name=selected_document,
        k_context=3,
    )

    print("LLM RESPONSE IS ", response["output_text"])

    # Accept a request from the frontend. Return the result from the search functions
    results = {
        "question": q,
        "answer": response["output_text"],
        "references": response["input_documents"],
        "selected_document": selected_document,
    }

    return results


class Feedback(BaseModel):
    rating: Union[str, int] = Field(
        description="""Recorded rating of the last answer.
        If thumbs are used then values are '1' for thumbs up
        and '0' for thumbs down."""
    )
    rating_comment: Optional[str] = Field(
        description="""Recorded comment on the last answer. Optional."""
    )
    question: Optional[str] = Field(description="""Last question. Optional.""")
    content_type: Optional[str] = Field(description="""Last content type. Optional.""")
    answer: Optional[str] = Field(description="""Last answer. Optional.""")


@app.post("/feedback", status_code=202, tags=["Principle Endpoints"])
async def record_rating(feedback: Feedback):
    """Records feedback on a previous answer.

    Args:
        feedback (Feedback): Recorded rating of the last answer.
            Required fields: rating (str or int).
            Optional fields: question, content_type, answer.

    Raises:
        HTTPException: 422 Validation error.

    Returns:
        HTTPResponse: 202 with empty body to indicate successfully added feedback.
    """
    logger.info(f"Recorded answer feedback: {feedback}")
    return ""
