import asyncio
from datetime import datetime
from typing import Any, Dict, List, Callable
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import os
# import tqdm.asyncio
from queue import Queue

from langchain.agents import AgentExecutor
from langchain.tools.base import ToolException
from transformers.agents.default_tools import Tool
from transformers.agents.agents import AgentError
from .evaluation.hard_questions import HARD_QUESTIONS

def acall_langchain_agent(agent: AgentExecutor, question: str) -> str:
    return agent.ainvoke({"input": question})

def call_langchain_agent(agent: AgentExecutor, question: str) -> str:
    return agent.invoke({"input": question})

async def arun_agent(
    example: Dict,
    agent_executor: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
    writer_queue: Queue = None,
    **kwargs
) -> dict:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    augmented_question = example["augmented_question"]
    try:
        # run executor agent
        response = await agent_call_function(agent_executor, augmented_question, **kwargs)

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step
                    for step in response["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in response["output"]
            else False
        )
        raised_exception = False

    except (ValueError, ToolException) as e:
        print("Error on ", augmented_question, e)
        response = {"output": None, "intermediate_steps": None}
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    intermediate_steps = response["intermediate_steps"]
    annotated_example = {
        "agent_name": agent_name,
        "question": example['question'],
        "augmented_question": augmented_question,
        "prediction": response["output"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "true_answer": example["true_answer"],
    }
    if writer_queue:
        writer_queue.put(annotated_example)
    return annotated_example


def run_agent(
    question: str,
    agent_executor: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
) -> dict:
    """
    Runs the execution process for a given question and ground truth answer.

    Args:
        question (str): The input question to be evaluated.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        agent_name (str): The name of the agent model.

    Returns:
        dict: A dictionary containing the evaluation results, including the agent model ID, evaluator model ID,
        question, ground truth answer, prediction, intermediate steps, evaluation score, evaluation feedback,
        tool call parsing error flag, iteration limit exceeded flag, and agent error (if any).
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # run executor agent
        response = agent_call_function(agent_executor, question)

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step[0].log
                    for step in response["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in response["output"]
            else False
        )
        raised_exception = False

    except Exception as e:
        response = {"output": None, "intermediate_steps": None}
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # collect results
    if response["intermediate_steps"] is not None:
        intermediate_steps = [
            {
                "tool": response[0].tool,
                "tool_input": response[0].tool_input,
                "tool_output": response[1],
            }
            for response in response["intermediate_steps"]
        ]
    else:
        intermediate_steps = None
    return {
        "agent_name": agent_name,
        "question": question,
        "prediction": response["output"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": repr(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }


def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


async def answer_questions(
    dataset: Dataset,
    agent: AgentExecutor,
    agent_name: str,
    output_folder: str = "output",
    agent_call_function: Callable = call_langchain_agent,
    visual_inspection_tool: Tool = None,
    text_inspector_tool: Tool = None,
    skip_hard_questions: bool = False
) -> List[Dict[str, Any]]:
    """
    Evaluates the agent on a given dataset.

    Args:
        dataset (Dataset): The dataset to test the agent on.
        agent: The agent.
        agent_name (str): The name of the agent model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each example in the dataset.
        Each dictionary includes the agent model ID, evaluator model ID, question, ground truth answer, prediction,
        intermediate steps, evaluation score, evaluation feedback, tool call parsing error flag, iteration limit
        exceeded flag, agent error (if any), and example metadata (task).
    """
    output_path = f"{output_folder}/{agent_name}.jsonl"
    print(f"Loading answers from {output_path}...")
    try:
        results = pd.read_json(output_path, lines=True).to_dict(orient="records")
        print(f"Found {len(results)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("Found no usable records! 🤔 Starting new.")
        results = []

    results_df = pd.DataFrame(results)

    for _, example in tqdm(enumerate(dataset), total=len(dataset)):
        if len(results_df) > 0:
            if example["question"] in results_df["question"].unique():
                continue
            if skip_hard_questions:
                if example["question"] in HARD_QUESTIONS:
                    continue
        prompt_use_files = ""
        if example['file_name']:
            if '.MOV' in example['file_name']:
                continue
            prompt_use_files += f"\n\nTo answer the question above, you will have to use these attached files:"
            if example['file_name'].split('.')[-1] in ['pdf', 'xlsx']:
                image_path = example['file_name'].split('.')[0] + '.png'
                if os.path.exists(image_path):
                    prompt_use_files += f"\nAttached image: {image_path}"
                else:
                    prompt_use_files += f"\nAttached file: {example['file_name']}"
            elif example['file_name'].split('.')[-1] == "zip":
                import shutil

                folder_name = example['file_name'].replace(".zip", "")
                os.makedirs(folder_name, exist_ok=True)
                shutil.unpack_archive(example['file_name'], folder_name)

                # Convert the extracted files
                prompt_use_files = "\n\nYou have been given a zip archive of supporting files. We extracted it into a directory: find the extracted files at the following paths:\n"
                for root, dirs, files in os.walk(folder_name):
                    for file in files:
                        file_path = os.path.join(root, file)
                        prompt_use_files += f"- {file_path}\n"
                        if file.split('.')[-1] in ['png', 'jpg', 'jpeg'] and visual_inspection_tool is not None:
                            prompt = f"""Write a caption of 5 sentences maximum for this image. Pay special attention to any details that might be useful for someone answering the following question:
{example['question']}. But do not try to answer the question directly!
Do not add any information that is not present in the image.
""".strip()
                            prompt_use_files += "> Description of this image: " + visual_inspection_tool(image_path=file_path, question=prompt) + '\n\n'
                        else:
                            prompt = f"""Write a short caption (5 sentences maximum) for this file. Pay special attention to any details that might be useful for someone answering the following question:
{example['question']}. But do not try to answer the question directly!
Do not add any information that is not present in the file.
""".strip()
                            prompt_use_files += "> Description of this file: " + text_inspector_tool(file_path=file_path, question=prompt, initial_exam_mode=True) + '\n\n'
            elif example['file_name'].split('.')[-1] in ['png', 'jpg', 'jpeg']:
                prompt_use_files += f"\nAttached image: {example['file_name']}"
            elif example['file_name'].split('.')[-1] in ['mp3', 'm4a', 'wav']:
                prompt_use_files += f"\nAttached audio: {example['file_name']}"
            else:
                prompt_use_files += f"\nAttached file: {example['file_name']}"

            if example['file_name'].split('.')[-1] in ['png', 'jpg', 'jpeg'] and visual_inspection_tool is not None:
                prompt = f"""Write a caption of 5 sentences maximum for this image. Pay special attention to any details that might be useful for someone answering the following question:
{example['question']}. But do not try to answer the question directly!
Do not add any information that is not present in the image.
""".strip()
                prompt_use_files += "\n> Description of this image: " + visual_inspection_tool(image_path=example['file_name'], question=prompt)
            elif '.zip' not in example['file_name'] and text_inspector_tool is not None:
                prompt = f"""Write a short caption (5 sentences maximum) for this file. Pay special attention to any details that might be useful for someone answering the following question:
{example['question']}. But do not try to answer the question directly!
Do not add any information that is not present in the file.
""".strip()
                prompt_use_files += "\n> Description of this file: " + text_inspector_tool(file_path=example['file_name'], question=prompt, initial_exam_mode=True)
        else:
            prompt_use_files += "\n\nYou have been given no local files to access."
        example['augmented_question'] = f"""It is paramount that you complete this task and provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it. Failure or 'I cannot answer' will not be tolerated, success will be rewarded.
Here is the task:
""" + example['question'] + prompt_use_files

        # run agent
        result = await arun_agent(
            example=example,
            agent_executor=agent,
            agent_name=agent_name,
            agent_call_function=agent_call_function,
        )

        # add in example metadata
        result.update(
            {
                "true_answer": example["true_answer"],
                "task": example["task"],
            }
        )
        results.append(result)

        with open(output_path, 'w') as f:
            for d in results:
                json.dump(d, f, default=serialize_agent_error)
                f.write('\n')  # add a newline for JSONL format
        # except Exception as e:
        #     print("EXCEPTION!!!!=================\nFIND THE EXCEPTION LOG BELOW:\n", e)
    return results


async def run_full_tests(
    dataset: Dataset,
    agents: Dict[str, AgentExecutor],
    agent_call_function: Callable = acall_langchain_agent,
    output_folder: str = "output",
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.

    Args:
        dataset (Dataset): The dataset to test on.
        agents (Dict[str, AgentExecutor]): A dictionary of agent executors to test on the dataset

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """
    results = []

    tasks = [
        answer_questions(
            dataset=dataset,
            agent=agent_executor,
            agent_name=agent_name,
            agent_call_function=agent_call_function,
            output_folder=output_folder,
        )
        for agent_name, agent_executor in agents.items()
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame([element for sublist in results for element in sublist])
