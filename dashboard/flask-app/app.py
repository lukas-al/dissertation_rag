import logging
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    make_response,
)
from flask.logging import default_handler
from markupsafe import escape
import requests


# statschat-api endpoint
endpoint = "http://localhost:8082/"  # TODO: add to some params/secrets file

# define session_id that will be used for log file and feedback
SESSION_NAME = f"statschat_app_{format(datetime.now(), '%Y_%m_%d_%H:%M')}"

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    # filename=f"log/{SESSION_NAME}.log",
    filemode="a",
)
logger.addHandler(default_handler)

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"


@app.route("/")
def home():
    """
    Home route for the Flask application.

    This function handles the home route of the application. It checks for the
    "latest_filter" parameter in the request arguments and updates the session
    accordingly. It then fetches a list of documents from an external endpoint
    and updates the session with the selected document.

    Returns:
        Response: A Flask response object with the rendered template or other
        appropriate content.
    """
    if "latest_filter" in request.args:
        session["latest_filter"] = request.args.get("latest_filter")
    else:
        session["latest_filter"] = "on"

    # Get our document list:
    response = requests.get(url=endpoint + "get_doc_list")
    doc_data = response.json()

    try:
        session["selected_document"]
    except KeyError:
        print("KEY ERROR IN FLASK")
        session["selected_document"] = "Please select a document from the list."

    if session["selected_document"] == "None":
        session["selected_document"] = "Please select a document from the list."

    browser_response = make_response(
        render_template(
            # "statschat.html",
            "srag_result.html",
            latest_filter=session["latest_filter"],
            question="",
            document_data=doc_data,
            selected_document=session["selected_document"],
        )
    )

    return browser_response


@app.route("/selectDocument", methods=["POST"])
def select_document_post():
    """
    Handles the POST request to select a document.

    This function retrieves the selected document from the form data,
    stores it in the session, and then redirects the user back to the home page.

    Returns:
        werkzeug.wrappers.Response: A redirect response to the home page.
    """
    # Get the arguments from the form
    selected_document = request.form.get("selected_document")
    session["selected_document"] = selected_document

    # Redirect back to home with the selected document now filled in
    return redirect(url_for("home"))


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Handles the search functionality for the application.

    This function retrieves the user's search query from the request arguments,
    cleans up session flags, fetches the document list from an external API,
    and sends the search query to another API endpoint to get the search results
    from the inquirer.

    Returns:
        werkzeug.wrappers.Response: The response object containing the search results.
    """
    # Get inputs from the user
    session["question"] = escape(request.args.get("q")).strip()

    # Clean up the flags
    session["content_type"] = "all"
    session["latest_filter"] = "off"

    # Fill in the document selector
    response = requests.get(url=endpoint + "get_doc_list")
    doc_data = response.json()

    # Send the question to the API and get it back
    if session["question"]:
        response = requests.get(
            url=endpoint + "search",
            params={
                "q": session["question"],
                "content_type": session["content_type"],
                "selected_document": session["selected_document"],
            },
        )

        session["answer"] = response.json()["answer"]
        docs = response.json()["references"]
        results = {"answer": session["answer"], "references": docs}

    else:
        results = {}

    # Fill in our response
    browser_response = make_response(
        render_template(
            # "statschat.html",
            "srag_result.html",
            latest_filter=session["latest_filter"],
            question=session["question"],
            results=results,
            document_data=doc_data,
            selected_document=session["selected_document"],
        )
    )

    return browser_response


@app.route("/record_rating", methods=["POST"])
def record_rating():
    """
    Handles the POST request to record a rating for the provided answer.

    This function retrieves the rating and comment from the form data,
    constructs a feedback payload including the question, content type,
    and answer from the session, and sends this feedback to an external
    API endpoint. It also logs the feedback for auditing purposes.

    Returns:
        tuple: An empty response with HTTP status code 204 indicating success.
    """
    rating = request.form["rating"]
    last_answer = {
        "rating": rating,
        "rating_comment": request.form["comment"],
        "question": session["question"],
        "content_type": session["content_type"],
        "answer": session["answer"],
    }
    requests.post(endpoint + "feedback", json=last_answer)
    logger.info(f"FEEDBACK: {last_answer}")
    return "", 204  # Return empty response with status code 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
