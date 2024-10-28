from transformers.agents.llm_engine import MessageRole, get_clean_message_list, llama_role_conversions
import os
from openai import OpenAI
from anthropic import Anthropic, AnthropicBedrock

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[], grammar=None):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
            response_format=grammar
        )
        return response.choices[0].message.content

class NIMEngine:
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model = model
        self.client = OpenAI(
            base_url="https://huggingface.co/api/integrations/dgx/v1",
            api_key=os.getenv("HF_NIM_TOKEN")
        )

    def __call__(self, messages, stop_sequences = [], grammar=None):
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop_sequences,
            max_tokens=1000,
        )
        return response.choices[0].message.content

class AnthropicEngine:
    def __init__(self, model_name=None, use_bedrock=False):
        self.model_name = model_name
        if use_bedrock: # Cf this page: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
            if model_name is None:
                model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            self.model_name = model_name
            self.client = AnthropicBedrock(
                aws_access_key=os.getenv("AWS_BEDROCK_ID"),
                aws_secret_key=os.getenv("AWS_BEDROCK_KEY"),
                aws_region="us-east-1",
            )
        else:
            if model_name is None:
                model_name = "claude-3-5-sonnet-20241022"
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        index_system_message, system_prompt = None, None
        for index, message in enumerate(messages):
            if message["role"] == MessageRole.SYSTEM:
                index_system_message = index
                system_prompt = message["content"]
        if system_prompt is None:
            raise Exception("No system prompt found!")

        filtered_messages = [message for i, message in enumerate(messages) if i!= index_system_message]
        if len(filtered_messages) == 0:
            print("Error, no user message:", messages)
            assert False

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=filtered_messages,
            stop_sequences=stop_sequences,
            temperature=0.5,
            max_tokens=2000
        )
        full_response_text = ""
        for content_block in response.content:
            if content_block.type == 'text':
                full_response_text += content_block.text
        return full_response_text


# import boto3
# import json
# import os

# class AnthropicBedrockEngine:
#     def __init__(self, model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"):
#         self.model_name = model_name
#         self.bedrock_runtime = boto3.client(
#             service_name='bedrock-runtime',
#             region_name='us-east-1',
#             aws_access_key_id=os.getenv("AWS_BEDROCK_ID"),
#             aws_secret_access_key=os.getenv("AWS_BEDROCK_KEY")
#         )


#     def __call__(self, messages, stop_sequences=[]):
#         messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
#         index_system_message, system_prompt = None, None
#         for index, message in enumerate(messages):
#             if message["role"] == MessageRole.SYSTEM:
#                 index_system_message = index
#                 system_prompt = message["content"]
#         if system_prompt is None:
#             raise Exception("No system prompt found!")

#         filtered_messages = [message for i, message in enumerate(messages) if i!= index_system_message]
#         if len(filtered_messages) == 0:
#             print("Error, no user message:", messages)
#             assert False

#         request = json.dumps({
#             "anthropic_version": "bedrock-2023-05-31",    
#             "max_tokens": 2000,
#             "system": system_prompt,    
#             "messages": filtered_messages,
#             "temperature": 0.5,
#             "stop_sequences": stop_sequences
#         })
#         response = self.bedrock_runtime.invoke_model(body=request, modelId=self.model_name)
#         response_body = json.loads(response.get('body').read())
    
#         full_response_text = ""
#         return full_response_text