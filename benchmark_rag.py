"""
A simplified implementation of "Benchmarking Large Language Models in 
Retrieval-Augmented Generation" by Chen et al (2023).
"""

import argparse
import warnings
from typing import Dict, List, Union

from transformers import pipeline

warnings.filterwarnings("ignore")

# Load the tiny llama model
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load the system and user prompts
with open("system_prompt.txt", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

with open("user_prompt_template.txt", "r", encoding="utf-8") as template:
    user_prompt_template = template.read()

# Define template for the prompt
prompt = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": user_prompt_template,
    },
]


def query_tiny_llama(
    question: str, document_path: str, messages: List[Dict[str, str]]
) -> str:
    """
    This function is used to query the tiny llama model.

    Args:
        question (str): Question to ask the model.
        document_path (str): Path to the document to use as context.

    Returns:
        str: Response for tiny llama model.
    """
    with open(document_path, "r", encoding="utf-8") as context_file:
        external_docs = context_file.read()
    messages[1]["content"] = messages[1]["content"].format(
        DOCS=external_docs,
        QUERY=question,
    )
    final_prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipe(
        final_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"]


def evaluate_accuracy(
    ground_truth: Union[str, List[str]],
    llama_response: str,
) -> bool:
    """
    This function is to used to evaluate if the response from tiny llama is
    accurate or not.

    Args:
        ground_truth (Union[str, List[str]]): Actual answer for the question.
        llama_response (str): Response from tiny llama.

    Returns:
        bool: True if the response is correct, else False.
    """
    llama_response = llama_response.lower()
    if isinstance(ground_truth, list):
        ground_truth = [value.lower() for value in ground_truth]
        return all(value in llama_response for value in ground_truth)
    ground_truth = ground_truth.lower()
    return ground_truth in llama_response


def evaluate_rejection_rate(llama_response: str) -> bool:
    """
    This function is to used to evaluate if the response from tiny llama was
    rejected or not.

    Args:
        llama_response (str): Response from tiny llama.

    Returns:
        bool: True if the response was rejected, else False.
    """
    llama_response = llama_response.lower()
    return "insufficient information" in llama_response


def evaluate_error_detection_rate(llama_response: str) -> bool:
    """
    This function is to used to evaluate if the response from tiny llama detect
    errors or not.

    Args:
        llama_response (str): Response from tiny llama.

    Returns:
        bool: True if the error was detected, else False.
    """
    llama_response = llama_response.lower()
    return "factual errors" in llama_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simplified version of Benchmarking Large Language Models in "
            "Retrieval-Augmented Generation by Chen et al."
        )
    )
    # get the test bed to be used in the benchmark.
    parser.add_argument(
        "--testbed",
        required=True,
        choices=[
            "noise_robustness",
            "negative_rejection",
            "information_integration",
            "counterfactual_robustness",
        ],
        help="Choose the testbed to run the benchmark.",
    )
    args = parser.parse_args()
    match args.testbed:
        case "noise_robustness":
            # for noise robustness we are interested in the accuracy alone.
            response = query_tiny_llama(
                "Who was awarded the 2022 Nobel prize in literature?",
                f"{args.testbed}.txt",
                prompt,
            )
            assistant_response = response.split("<|assistant|>")[1]
            correctness = evaluate_accuracy("Annie Ernaux", assistant_response)
            print(f"Details of Experiment:\n{response}\n")
            print(
                f"Tiny llama was {'correct' if correctness else 'incorrect'}."
            )
        case "negative_rejection":
            # for negative rejection we are interested in the rejection rate.
            response = query_tiny_llama(
                "Who was awarded the 2022 Nobel prize in literature?",
                f"{args.testbed}.txt",
                prompt,
            )
            assistant_response = response.split("<|assistant|>")[1]
            rejection = evaluate_rejection_rate(assistant_response)
            print(f"Details of Experiment:\n{response}\n")
            print(f"Tiny llama {'rejected' if rejection else 'didn`t reject'}.")
        case "information_integration":
            # for information integration we are interested in the accuracy only.
            response = query_tiny_llama(
                "When were the ChatGPT app for iOS and ChatGPT api launched?",
                f"{args.testbed}.txt",
                prompt,
            )
            assistant_response = response.split("<|assistant|>")[1]
            correctness = evaluate_accuracy(
                ["May 18", "March 1"], assistant_response
            )
            print(f"Details of Experiment:\n{response}\n")
            print(
                f"Tiny llama was {'correct' if correctness else 'incorrect'}."
            )
        case "counterfactual_robustness":
            # for counterfactual robustness we are interested in the accuracy
            # and error detection rate.
            response = query_tiny_llama(
                "Which city hosted the Olympic games in 2004?",
                f"{args.testbed}.txt",
                prompt,
            )
            assistant_response = response.split("<|assistant|>")[1]
            correctness = evaluate_accuracy("Athens", assistant_response)
            detection = evaluate_error_detection_rate(assistant_response)
            print(f"Details of Experiment:\n{response}\n")
            print(
                f"Tiny llama was {'correct' if correctness else 'incorrect'} "
                f"and {'detected' if detection else 'didn`t detect'} the "
                "factual inconsistency."
            )
        case _:
            print("Invalid testbed.")
    print("Adios!✌🏾")
