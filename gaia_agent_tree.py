# THIS IS EXPERIMENTAL! IT RELIES ON A NON-YET-MERGED BRANCH OF TRANFORMERS? IT WONT WORK OUT OF THE BOX.


import asyncio
import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import datasets
from huggingface_hub import login
from transformers.agents import ReactCodeAgent, ReactJsonAgent, HfApiEngine
from transformers.agents.agents import DEFAULT_REACT_JSON_SYSTEM_PROMPT
from transformers.agents.default_tools import Tool, PythonInterpreterTool
from transformers.agents.llm_engine import MessageRole
from scripts.tools.web_surfer import (
    SearchInformationTool,
    NavigationalSearchTool,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
)
from scripts.tools.mdconvert import MarkdownConverter
from scripts.reformulator import prepare_response
from scripts.run_agents import answer_questions
from scripts.tools.visual_qa import VisualQATool, VisualQAGPT4Tool
from scripts.llm_engines import OpenAIEngine, AnthropicEngine, NIMEngine
load_dotenv(override=True)
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

OUTPUT_DIR = "output"
USE_OS_MODELS = False

SET = "validation"

# proprietary_llm_engine = AnthropicEngine(use_bedrock=True)
proprietary_llm_engine = OpenAIEngine()

repo_id_llama3 = "meta-llama/Meta-Llama-3-70B-Instruct"
repo_id_command_r = "CohereForAI/c4ai-command-r-plus"
repo_id_gemma2 = "google/gemma-2-27b-it"
repo_id_llama = "meta-llama/Meta-Llama-3.1-70B-Instruct"

REPO_ID_OS_MODEL = repo_id_llama
### LOAD EVALUATION DATASET

eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
eval_ds = eval_ds.rename_columns(
    {"Question": "question", "Final answer": "true_answer", "Level": "task"}
)


def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
    return row


eval_ds = eval_ds.map(preprocess_file_paths)

eval_df = pd.DataFrame(eval_ds)
print("Loaded evaluation dataset:")
print(pd.Series(eval_ds["task"]).value_counts())


websurfer_llm_engine = NIMEngine(
    model=REPO_ID_OS_MODEL,
)  # chosen for its high context length

# Replace with OAI if needed
if not USE_OS_MODELS:
    websurfer_llm_engine = proprietary_llm_engine

### BUILD AGENTS & TOOLS

WEB_TOOLS = [
    SearchInformationTool(),
    NavigationalSearchTool(),
    VisitTool(),
    PageUpTool(),
    PageDownTool(),
    FinderTool(),
    FindNextTool(),
    ArchiveSearchTool(),
]

class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "text",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!",
            "type": "text",
        },
    }
    output_type = "text"
    md_converter = MarkdownConverter()

    def forward(self, file_path, question: Optional[str] = None, initial_exam_mode: Optional[bool] = False) -> str:

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in ['.png', '.jpg']:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content
        
        if not question:
            return result.text_content
        
        if initial_exam_mode:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": "Here is a file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": MessageRole.USER,
                    "content": question,
                },
            ]
            return websurfer_llm_engine(messages)
        else:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": "You will have to write a short caption for this file, then answer this question:"
                    + question,
                },
                {
                    "role": MessageRole.USER,
                    "content": "Here is the complete file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": MessageRole.USER,
                    "content": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                    + question,
                },
            ]
            return websurfer_llm_engine(messages)


surfer_agent = ReactJsonAgent(
    llm_engine=websurfer_llm_engine,
    tools=WEB_TOOLS,
    max_iterations=10,
    verbose=2,
    # grammar = DEFAULT_JSONAGENT_REGEX_GRAMMAR,
    system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT,
    planning_interval=4,
    plan_type="default",
)

from transformers.agents.agents import ManagedAgent

search_agent = ManagedAgent(
    surfer_agent,
    "web_search",
    description="""A team member that will browse the internet to answer your question.
Ask him for all your web-search related questions, but he's unable to do problem-solving.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.""",
    additional_prompting="""You can navigate to .txt or .pdf online files using your 'visit_page' tool.
If it's another format, you can return the url of the file, and your manager will handle the download and inspection from there.
Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
""",
    provide_run_summary=True
)

ti_tool = TextInspectorTool()

TASK_SOLVING_TOOLBOX = [
    VisualQAGPT4Tool(),  # VisualQATool(),
    ti_tool,
]

hf_llm_engine = NIMEngine(model=REPO_ID_OS_MODEL)

llm_engine = hf_llm_engine if USE_OS_MODELS else proprietary_llm_engine

react_agent = ReactCodeAgent(
    llm_engine=llm_engine,
    tools=TASK_SOLVING_TOOLBOX,
    max_iterations=12,
    verbose=0,
    # grammar=DEFAULT_CODEAGENT_REGEX_GRAMMAR,
    additional_authorized_imports=[
        "requests",
        "zipfile",
        "os",
        "pandas",
        "numpy",
        "sympy",
        "json",
        "bs4",
        "pubchempy",
        "xml",
        "yahoo_finance",
        "Bio",
        "sklearn",
        "scipy",
        "pydub",
        "io",
        "PIL",
        "chess",
        "PyPDF2",
        "pptx",
        "torch",
        "datetime",
        "csv",
        "fractions",
    ],
    # planning_interval=3,
    managed_agents=[search_agent]
)

managed_solver_agent_1 = ManagedAgent(
    react_agent,
    "solver_agent_1",
    description="""A team member that will try to solve the task given to you. Make sure to give the main task exactly as it was given to you, word for word, to both this first solver agent first, then also give it to your second agent. Then you will be able to compare answers to decide a final answer.""",
    additional_prompting="""Make sure to give a correct answer!""",
    provide_run_summary=True
)

managed_solver_agent_2 = ManagedAgent(
    react_agent,
    "solver_agent_2",
    description="""A team member that will try to solve the task given to you. Make sure to give the main task exactly as it was given to you, word for word, to both this first solver agent first, then also give it to your second agent. Then you will be able to compare answers to decide a final answer.""",
    additional_prompting="""You need to put all your effort into giving an exact answer!""",
    provide_run_summary=True
)

orchestrator_agent = ReactCodeAgent(
    llm_engine=llm_engine,
    tools=[],
    max_iterations=4,
    managed_agents=[managed_solver_agent_1, managed_solver_agent_2]
)

### EVALUATE

async def call_transformers(agent, question: str, **kwargs) -> str:
    question += "\n---\nCall both your solver agents WITH THE SAME EXACT TASK THAT WAS GIVEN TO YOU ABOVE, INCLUDING ALL EXACT DESCRIPTIONS OF POTENTIAL ATTACHED FILES, then compare results. If insure you can run them again."
    result = agent.run(question, **kwargs)
    agent_memory = agent.write_inner_memory_from_logs(summary_mode=True)
    try:
        final_result = prepare_response(question, agent_memory, llm_engine)
    except Exception as e:
        print(e)
        final_result = result
    return {
        "output": str(final_result),
        "intermediate_steps": [
            {key: value for key, value in log.items() if key != "agent_memory"}
            for log in agent.logs
        ],
    }


results = asyncio.run(answer_questions(
    eval_ds,
    orchestrator_agent,
    "gpt-4o_03_sept_orchestrator",
    output_folder=f"{OUTPUT_DIR}/{SET}",
    agent_call_function=call_transformers,
    visual_inspection_tool = VisualQAGPT4Tool(),
    text_inspector_tool = ti_tool,
))