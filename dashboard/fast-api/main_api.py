from pydantic import BaseModel, Field
from typing import Union, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging
from datetime import datetime
from markupsafe import escape

from StructuredRag.algorithms.inquirer import StructRAGInquirer

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    # filename=f"log/{SESSION_NAME}.log",
    filemode="a",
)

inquirer = StructRAGInquirer(
    # path_to_experiment='/Users/lukasalemu/Documents/00. Bank of England/00. Degree/Dissertation/structured-rag/results/v0/2024-05-28',
    path_to_experiment=r'N:\CECD\10. Personal\Lukas Alemu\Study Repository\99. Capstone\dissertation_rag\results\v0\2024-05-28',
    llm_name='google/flan-t5-large',
    llm_max_tokens=512,
)
    
app = FastAPI(
    title="Structured RAG API",
    description="Experimental search using Structured RAG for the sample data",
    summary="""Experimental search of MA Notes and Data publications.
        Using retrieval augmented generation (RAG).""",
    version="0.0.2",
    contact={
        "name": 'Lukas Alemu',
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
    print('Selected doc is', selected_document)  
    response = inquirer.run_inquirer(
        query=q,
        source_document_name=selected_document,
        k_context=3,
    )
    
    print("LLM RESPONSE IS ", response['output_text'])
    
    # Accept a request from the frontend. Return the result from the search functions
    results = {
        "question": q,
        "answer": response['output_text'],
        "references": response['input_documents'],
        "selected_document": selected_document,
    }
    
    return results


# @app.get("/search", tags=["Principle Endpoints"])
# async def search(
#     q: str,
#     content_type: Union[str, None] = "latest",
#     debug: bool = True,
# ):
#     """Search ONS articles and bulletins for a question.

#     Args:
#         q (str): Question to be answered based on ONS articles and bulletins.
#         content_type (Union[str, None], optional): Type of content to be searched.
#             Currently accepted values: 'latest' to search the latest bulletins only
#             or 'all' to search any articles and bulletins.
#             Optional, defaults to 'latest'.
#         debug (bool, optional): Flag to return debug information (full LLM response).
#             Optional, defaults to True.

#     Raises:
#         HTTPException: 422 Validation error.

#     Returns:
#         HTTPresponse: 200 JSON with fields: question, content_type, answer, references
#             and optionally debug_response.
#     """
#     question = escape(q).strip()
#     if question in [None, "None", ""]:
#         raise HTTPException(status_code=422, detail="Empty question")

#     if content_type not in ["latest", "all"]:
#         logger.warning('Unknown content type. Fallback to "latest".')
#         content_type = "latest"
#     latest_weight = get_latest_flag({"q": question}, CONFIG["app"]["latest_max"])

#     docs, answer, response = inquirer.make_query(
#         question,
#         latest_filter=content_type == "latest",
#         latest_weight=latest_weight,
#     )
#     results = {
#         "question": question,
#         "content_type": content_type,
#         "answer": answer,
#         "references": docs,
#     }
#     if debug:
#         results["debug_response"] = response.__dict__
#     logger.info(f"Sending following response: {results}")
#     return results


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
