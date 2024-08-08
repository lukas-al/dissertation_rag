"""Using LLM to perform evaluation chain"""

from tqdm.auto import tqdm
from typing import List
import json
import pickle
import pathlib
import random
import cleantext
from llama_cpp import Llama


from StructuredRag.utils import mistral_conversation
from StructuredRag.algorithms.inquirer import StructRAGInquirer

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent

# Shortened and fitting into the CHATML format
# QA_SYSTEM_PROMPT = """ 
# Your task is to write a factoid question and an answer given a context.
# Your factoid question should be answerable with a specific, concise piece of factual information from the context.

# Provide your answer as follows:

# Output:::
# Factoid question: (your factoid question)
# Answer: (your answer to the factoid question)
# """

QA_SYSTEM_PROMPT = """ 
Your task is to write a meaningful question and an answer given a context.
Your question should be answerable using information which is present in the context. It should be both open-ended and also ask for specific, concise information from the context.
The question should be useful to a macroeconomist working at the Bank of England.

Provide your answer as follows:

Output:::
Question: (your question)
Answer: (your answer to the question)
"""


QA_GROUNDEDNESS_PROMPT = """
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the context and question.
"""

QA_RELEVANCE_PROMPT = """
Your task is to provide a 'total rating' representing how useful this question can be to macro-economists looking for information whilst working at the Bank of England.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.
"""

QA_STANDALONE_PROMPT = """
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like MPC, CPI or YBUS and still be a 5.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.
"""

JUDGE_PROMPT = """
###Task Description:
An instruction (including the context) a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
You are an AI assistant with a focus on helping to answer economists' search questions over particular documents. 
Respond only to the question asked, the response should be concise and relevant, and use the context provided to give a comprehensive answer.
It is important to maintain impartiality and non-partisanship. If you are unable to answer a question based on the given instructions and context, please indicate so.
Your responses should be well-structured and professional, using British English.

{query} Use the following context to answer the question:
Context: {context}

###Response to evaluate:
{answer_RAG_system}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback: 
"""


def build_context_for_QA_gen(doc, rag_agent, k_context: int = 3):
    """
    Builds the context string for generating synthetic question-answering pairs.

    Args:
        doc: The document for which the context is being built.
        rag_agent: The RAG agent used for retrieving similar nodes.
        k_context (int): The number of similar nodes to consider for building the context. Default is 3.

    Returns:
        context_string (str): The generated context string containing the clean text of similar nodes.
    """
    similar_nodes = rag_agent._graph_similar_nodes(doc.id_, k_context)

    context_string = """ """
    for i, (node_id, _) in enumerate(similar_nodes):
        # Yes its unoptimised... find the document who's id matches the node_id
        node = next((x for x in embedded_index if x.id_ == node_id), None)

        clean_text = (
            node.text.replace("\n", " ").replace("\t", " ").replace("  ", " ").strip()
        )

        clean_text = cleantext.clean(clean_text)

        context_string += f"Context item {i}: {clean_text} \n"

    return context_string


def create_chatML_QA_prompt(document_text: str) -> List[dict]:
    """
    Create a chatML QA prompt for the given document text.

    Args:
        document_text (str): The text of the document.

    Returns:
        List[dict]: A list of dictionaries representing the chatML QA prompt.
            Each dictionary has two keys: 'role' and 'content'.
            The 'role' can be either 'system' or 'user'.
            The 'content' contains the text of the role.
    """
    return [
        {
            "role": "system",
            "content": QA_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": document_text,
        },
    ]


def create_chatML_quality_prompt(
    prompt: str, question: str, context: str = None
) -> List[dict]:
    """
    Creates a chatML quality prompt for factoid questions with optional context.

    Args:
        prompt (str): The prompt for the system role.
        question (str): The factoid question for the user role.
        context (str, optional): The context for the user role. Defaults to None.

    Returns:
        List[dict]: A list of dictionaries representing the chatML quality prompt.
    """
    if context is None:
        return [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"Factoid question: {question}",
            },
        ]
    else:
        return [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}\nFactoid question: {question}",
            },
        ]


if __name__ == "__main__":
    experiment_path = "v3/2024-08-07"
    mistral_model_path = pathlib.Path(
        r"C:\Users\335257\.cache\huggingface\hub\models--TheBloke--CapybaraHermes-2.5-Mistral-7B-GGUF\snapshots\234067be357852d0c75bf1d04d2c720d15eab3e2\capybarahermes-2.5-mistral-7b.Q5_0.gguf"
    )
    prometheus_model_path = pathlib.Path(
        r"C:\Users\335257\.cache\huggingface\hub\models--vsevolodl--prometheus-7b-v2.0-GGUF\snapshots\b0c3627c40a9ab93c358a110702d8e3a6b20e5d7\prometheus-7b-v2.0.Q5_K_M.gguf"
    )
    N_QA_PAIRS = 10
    
    with open(
        PROJECT_ROOT / "results" / experiment_path / "embedded_index.pickle", "rb"
    ) as f:
        embedded_index = pickle.load(f)

    
    inquirer = StructRAGInquirer(
        path_to_experiment=str(PROJECT_ROOT / "results" / experiment_path),
        llm_type="llamacpp",
        # model_path=str(PROJECT_ROOT / "capybarahermes-2.5-mistral-7b.Q5_K_M.gguf"),
        model_path=str(mistral_model_path),
        llm_max_tokens=1600,
        n_gpu_layers=0,  # All layers
        use_anchor_document=False,
        n_threads=4,
    )

    print(f"succeeded in loading inquirer: {inquirer}")

    raw_outputs = []
    outputs = []
    for sample_doc in tqdm(
        random.sample(embedded_index, N_QA_PAIRS), desc="Generating QA pairs"
    ):
        context_text = build_context_for_QA_gen(sample_doc, inquirer, k_context=3)
        try:
            output = inquirer.llm.create_chat_completion(
                messages=create_chatML_QA_prompt(context_text)
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                print(
                    f"Skipping sample document {sample_doc.id_} as it exceeds the token context window. \n Error: {e}"
                )
                continue

        raw_outputs.append(output)
        try:
            outputs.append(
                {
                    "context": context_text,
                    "question": output["choices"][0]["message"]["content"]
                    .split("Question: ")[1]
                    .split("\n")[0],
                    "answer": output["choices"][0]["message"]["content"].split(
                        "Answer: "
                    )[1],
                }
            )
        except IndexError as e:
            print(
                f"Skipping sample document {sample_doc.id_} as it failed to generate a question-answer pair. \n Error: {e}"
            )
            continue

    # Persist the intermediate outputs
    with open(
        PROJECT_ROOT / "results" / experiment_path / "qa_pairs_raw.json", "w"
    ) as f:
        json.dump(outputs, f, indent=4)

    for output_bundle in tqdm(outputs, desc="Evaluating QA pairs"):
        try:
            groundedness_eval = inquirer.llm.create_chat_completion(
                messages=create_chatML_quality_prompt(
                    QA_GROUNDEDNESS_PROMPT,
                    output_bundle["question"],
                    output_bundle["context"],
                ),
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                print(
                    f"Skipping sample document {sample_doc.id_} as it exceeds the token context window. \n Error: {e}"
                )
                continue

        try:
            relevance_eval = inquirer.llm.create_chat_completion(
                messages=create_chatML_quality_prompt(
                    QA_RELEVANCE_PROMPT, output_bundle["question"]
                ),
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                print(
                    f"Skipping sample document {sample_doc.id_} as it exceeds the token context window. \n Error: {e}"
                )
                continue

        try:
            standalone_eval = inquirer.llm.create_chat_completion(
                messages=create_chatML_quality_prompt(
                    QA_STANDALONE_PROMPT, output_bundle["question"]
                ),
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                print(
                    f"Skipping sample document {sample_doc.id_} as it exceeds the token context window. \n Error: {e}"
                )
                continue

        raw_outputs.append(
            {
                "groundedness": groundedness_eval,
                "relevance": relevance_eval,
                "standalone": standalone_eval,
            }
        )

        # Extract the scores and write them
        for eval_type, eval_output in zip(
            ["groundedness", "relevance", "standalone"],
            [groundedness_eval, relevance_eval, standalone_eval],
        ):
            # If the model has stopped generating text due to stopping itself, rather than max tokens, etc.
            if eval_output["choices"][0]["finish_reason"] == "stop":
                try:
                    output_bundle.update(
                        {
                            f"{eval_type}_score": eval_output["choices"][0]["message"][
                                "content"
                            ]
                            .split("Total rating: ")[1]
                            .split("\n")[0]
                            .strip(),
                            f"{eval_type}_rationale": eval_output["choices"][0][
                                "message"
                            ]["content"]
                            .split("Evaluation: ")[1]
                            .split("\n")[0]
                            .strip(),
                        }
                    )
                except IndexError as e:
                    print(
                        f"Skipping sample document {sample_doc.id_} as it failed to generate a {eval_type} evaluation. \n Error: {e}"
                    )
                    continue
            else:
                output_bundle.update(
                    {f"{eval_type}_score": "0", f"{eval_type}_rationale": ""}
                )

    # Persist the intermediate outputs
    with open(
        PROJECT_ROOT / "results" / experiment_path / "qa_pairs_evaluated.json", "w"
    ) as f:
        json.dump(outputs, f, indent=4)

    # Filter out any bad outputs using the scores - simple lambda with gets to avoid KeyError
    filtered_outputs = list(
        filter(
            lambda x: (
                float(x.get("groundedness_score", 0)) >= 2
                and float(x.get("relevance_score", 0)) >= 2
                and float(x.get("standalone_score", 0)) >= 2
            ),
            outputs,
        )
    )

    ### GENERATE ANSWERS TO THE QUESTIONS
    for qa_bundle in tqdm(filtered_outputs, desc="Running inquirer"):
        response = inquirer.run_inquirer(
            query=qa_bundle["question"],
            source_document_name=None,
            k_context=3,
        )

        raw_outputs.append(response)

        try:
            qa_bundle.update(
                {
                    "RAG_response": response,
                    "RAG_response_text": response["choices"][0]["message"]["content"],
                }
            )
        except IndexError as e:
            print(
                f"Skipping sample document {sample_doc.id_} as it failed to generate a response. \n Error: {e}"
            )
            continue

    # Persist the outputs
    with open(
        PROJECT_ROOT / "results" / experiment_path / "qa_pairs_answered.json", "w"
    ) as f:
        json.dump(filtered_outputs, f, indent=4)

    
    ### LOAD FOR PERSISTENCE -> READ FROM FILE
    with open(PROJECT_ROOT / "results" / experiment_path / "qa_pairs_answered.json", "r") as f:
        filtered_outputs = json.load(f)
    
    ### FINAL EVALUATION USING JUDGE LLM
    judge_llm = Llama(
        model_path=str(prometheus_model_path), verbose=False, n_gpu_layers=0, n_ctx=1750, n_threads=4,
    )

    conv = mistral_conversation.get_conv_template("mistral")
    conv.set_system_message(
        "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
    )
    conv.append_message(conv.roles[0], JUDGE_PROMPT)
    conv.append_message(conv.roles[1], None)
    judge_prompt = conv.get_prompt()

    for qa_bundle in tqdm(filtered_outputs, desc="Running judgement llm"):
        try:
            evaluation = judge_llm.create_completion(
                prompt=judge_prompt.format(
                    query=qa_bundle["question"],
                    context=qa_bundle["context"],
                    reference_answer=qa_bundle["answer"],
                    answer_RAG_system=qa_bundle["RAG_response"]["choices"][0]["message"][
                        "content"
                    ],
                ),
                echo=True,
                max_tokens=None,
            )
        except ValueError as e:
            if "exceed context window" in str(e):
                print(
                    f"Skipping sample document {sample_doc.id_} as it exceeds the token context window. \n Error: {e}"
                )
                qa_bundle.update(
                    {
                        "judge_evaluation": {"choices": [{"text": "0"}]},
                        "judge_score": "0",
                    }
                )
                continue

        qa_bundle.update(
            {
                "judge_evaluation": evaluation,
                "judge_score": evaluation["choices"][0]["text"]
                .split("[RESULT]")[-1]
                .strip(),
            }
        )

    # Persist the outputs
    with open(
        PROJECT_ROOT / "results" / experiment_path / "qa_pairs_judged.json", "w"
    ) as f:
        json.dump(filtered_outputs, f, indent=4)

    with open(
        PROJECT_ROOT / "results" / experiment_path / "qa_raw_output.json", "w"
    ) as f:
        json.dump(raw_outputs, f, indent=4)
