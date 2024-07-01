import os
import re
import json
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts.chat import ChatPromptTemplate
import pandas as pd
import asyncio
from typing import Optional, List, Union
import tqdm.asyncio
import numpy as np
from threading import Thread
from queue import Queue
import datasets

import numpy as np
import re


_SENTINEL_KILL_CONSUMERS = object()


def build_evaluator(hf_endpoint_url: str) -> tuple:
    """
    Build an evaluator language model using the given Hugging Face endpoint URL.

    Args:
        hf_endpoint_url (str): The URL of the Hugging Face endpoint.

    Returns:
        Tuple: A tuple containing the evaluator chat model and the correctness prompt template.
    """
    eval_chat_model = HuggingFaceEndpoint(
        endpoint_url=hf_endpoint_url,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 488,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )
    return eval_chat_model


async def evaluate_single_example(
    example: dict, evaluator, eval_prompt_template, evaluator_name, eval_split_string="[RESULT]", writer_queue: Optional[Queue] = None
):
    if f"eval_score_{evaluator_name}" in example:
        try:
            el = float(example[f"eval_score_{evaluator_name}"])
            assert not np.isnan(el)
            return example
        except:
            pass
    eval_prompt = eval_prompt_template.format_messages(
        instruction=example["question"],
        response=example["prediction"],
        reference_answer=example["true_answer"],
    )
    print("Evaluating example")
    eval_result = await evaluator.ainvoke(eval_prompt)
    eval_result = eval_result.content
    try:
        feedback, score = [item.strip() for item in eval_result.split(eval_split_string)]
    except:
        print(eval_result)
        segments = [
            segment.strip() for segment in eval_result.split(eval_split_string) if segment.strip()
        ]
        # Search for a segment that contains a numerical score
        for segment in segments:
            if segment.isdigit():
                feedback = ""
                score = int(segment)
    example[f"eval_score_{evaluator_name}"] = score
    example[f"eval_feedback_{evaluator_name}"] = feedback
    if writer_queue:
        writer_queue.put(example)
    return example


async def evaluate_answers(
    examples,
    evaluator,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
    eval_split_string: str = "[RESULT]",
    output_file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.
    Uses safe writing in multithreading, from options suggested here:
    https://stackoverflow.com/questions/33107019/multiple-threads-writing-to-the-same-csv-in-python

    Args:
        dataset (Dataset): The dataset to test on.
        agents (Dict[str, AgentExecutor]): A dictionary of agent executors to test on the dataset

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """
    examples_to_do = examples
    previous_evaluations = pd.DataFrame()
    
    if output_file_path and os.path.isfile(output_file_path):
        previous_evaluations = pd.read_json(output_file_path, lines=True)
        print(f'Found {len(previous_evaluations)} previous evaluations!')
        if f"eval_score_{evaluator_name}" in previous_evaluations.columns:
            previous_evaluations = previous_evaluations.loc[previous_evaluations[f"eval_score_{evaluator_name}"].notna()]
            
            examples_to_do = [example for example in examples if not len(previous_evaluations.loc[
                (previous_evaluations["question"] == example["question"]) & (previous_evaluations["agent_name"] == example["agent_name"])
            ]) > 0]

    print(f"Launching evaluation for {len(examples_to_do)} examples...")
    writer_queue = Queue()

    with open(output_file_path, "a") as output_file:
        def write_line():
            while True:
                if not writer_queue.empty():
                    annotated_example = writer_queue.get()
                    
                    if annotated_example is _SENTINEL_KILL_CONSUMERS:
                        writer_queue.put(_SENTINEL_KILL_CONSUMERS) # put it back so that other consumers see it
                        return
                    
                    annotated_example = {k: str(v) for k, v in annotated_example.items()}

                    # Row comes out of writer_queue; JSON writing goes here
                    json.dump(annotated_example, output_file)
                    output_file.write('\n')
        
        consumer = Thread(target=write_line)
        consumer.setDaemon(True)
        consumer.start()

        tasks = [
            evaluate_single_example(
                example,
                evaluator,
                eval_prompt_template,
                evaluator_name,
                eval_split_string,
                writer_queue,
            )
            for example in examples_to_do
        ]

        evaluation_results = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        writer_queue.put(_SENTINEL_KILL_CONSUMERS)

    return evaluation_results + previous_evaluations.to_dict(orient="records")


def extract_numbers(string):
    try:
        found_strings = [el.strip() for el in re.findall(r"(?:[,\d]+.?\d*)", string)]

        found_strings = [
            "".join(ch for ch in el if (ch.isalnum() or ch == "."))
            for el in found_strings
            if el[0].isdigit() or el[0] == "."
        ]
        found_strings = [float(el) for el in found_strings if len(el) > 0]

        return found_strings
    except Exception as e:
        print("Error when extracting string:", e)
        return 0


def split_answer(row):
    if row['task'] == 'GSM8K':
        splitted = row["true_answer"].split("####")
        row["true_reasoning"] = splitted[0]
        str_answer = splitted[1].strip().replace(",", "") # remove thousand separators from GSM8K
        row["true_answer"] = float(str_answer)
    return row


def load_math_datasets(n_eval_samples = 30):
    math_dataset = (
        datasets.load_dataset("GSM8K", "main")["train"].shuffle(seed=496).select(range(100))
    )
    math_dataset = pd.DataFrame(math_dataset)

    math_dataset = math_dataset.apply(split_answer, axis=1)
    math_dataset = math_dataset.drop(columns=["answer"]).iloc[:100]
    math_dataset = datasets.Dataset.from_pandas(math_dataset)

    return math_dataset


def load_benchmark():
    dataset = datasets.load_dataset("m-ric/agents_medium_benchmark")['train']
    dataset = dataset.rename_column("answer", "true_answer")
    df = pd.DataFrame(dataset)
    return df.apply(split_answer, axis=1)


def extract_numbers(output):
    if isinstance(output, float) or isinstance(output, int):
        return [output]
    try:
        found_strings = [el.strip() for el in re.findall(r"(?:[,\d]+.?\d*)", output)]

        found_strings = [
            "".join(ch for ch in el if (ch.isalnum() or ch == "."))
            for el in found_strings
            if el[0].isdigit() or el[0] == "."
        ]
        found_strings = [float(el) for el in found_strings if len(el) > 0]

        return found_strings

    except Exception as e:
        print("Error when extracting string:", e)
        return []


def score_any_match(prediction: str, true_answer: Union[str, int, float]) -> bool:
    """Scores if any number extracted from the prediction matches the true answer"""
    extracted_numbers = extract_numbers(prediction)
    found_match = any(
        [
            np.isclose(extracted_number, float(true_answer), atol=0.1, rtol=0.05)
            for extracted_number in extracted_numbers
        ]
    )
    return found_match

def score_last_match(prediction: str, true_answer: Union[str, int, float]) -> bool:
    """Scores if any number extracted from the prediction matches the true answer"""
    extracted_numbers = extract_numbers(prediction)
    if len(extracted_numbers) == 0:
        return False
    return np.isclose(extracted_numbers[-1], float(true_answer), atol=0.1, rtol=0.05)


def score_any_match_series(predictions: pd.Series, true_answers: pd.Series) -> List:
    return [score_any_match(predictions.values[i], true_answers.values[i]) for i in range(len(predictions.values))]

def score_last_match_series(predictions: pd.Series, true_answers: pd.Series) -> List:
    return [score_last_match(predictions.values[i], true_answers.values[i]) for i in range(len(predictions.values))]


def score_naive_match(prediction: str, true_answer: str):
    if len(prediction) <= len(true_answer):
        return prediction.lower() == true_answer.lower()
    else:  # find substring with highest score
        return any(
            [
                prediction[offset : offset + len(true_answer)].lower()
                == true_answer.lower()
                for offset in range(len(prediction) - len(true_answer))
            ]
        )
    
def is_number(am_i_a_number):
    return am_i_a_number.strip().lstrip('-').replace('.', '', 1).replace(',', '').isdigit()
