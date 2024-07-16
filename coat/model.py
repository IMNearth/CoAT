from typing import Union, List, Dict, Optional, Any
from abc import abstractmethod
import os
import io
import json
import requests
import random
import base64
import tempfile
import numpy as np
from PIL import Image
from functools import partial
from http import HTTPStatus


class BaseModel(object):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def format_query(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get_response(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def chat(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_response(*args, **kwargs)


class OpenAIModel(BaseModel):

    def __init__(self, configs, seed=2024, **kwargs) -> None:
        super().__init__()

        self.api_key = configs['OPENAI_API_KEY']
        self.api_url = configs['OPENAI_API_URL']
        self.seed = seed
    
    def encode_image(self, image=None, image_path=None):
        assert image is not None or image_path is not None, \
            "Please specify at least an image array or a path to image."
        if image is not None:# convert image to bytes
            with io.BytesIO() as output_bytes:
                if isinstance(image, Image.Image): PIL_image = image
                elif isinstance(image, np.ndarray): PIL_image = Image.fromarray(image)
                else: raise TypeError(f"Unsupported image type {type(image)}")
                PIL_image.save(output_bytes, 'PNG')
                bytes_data = output_bytes.getvalue()
            return base64.b64encode(bytes_data).decode('utf-8')
        elif image_path is not None:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else: raise NotImplementedError
    
    def format_query(self, image_paths:List[str], prompt:Union[str, dict, tuple], detail="high"):
        if isinstance(image_paths, str): image_paths = [image_paths]

        if isinstance(prompt, str):            
            message = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},]}]
            if len(image_paths) != 0:
                for img_path in image_paths:
                    base64_image = self.encode_image(image_path=img_path)
                    message[0]['content'].append(
                        {'type': 'image_url', 'image_url': {"url": f"data:image/jepg;base64,{base64_image}", "detail": detail}}
                    )
        
        if isinstance(prompt, list):
            message = []
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    base64_image = self.encode_image(image_path=image_paths[img_index])
                    message.append({'role': role, 'content': [
                        {'type': 'text', 'text': msg['text']},
                        {'type': 'image_url', 'image_url': {"url": f"data:image/jepg;base64,{base64_image}", "detail": detail}}
                    ]})
                else:
                    if isinstance(msg, dict):
                        message.append({'role': role, 'content': [{'type': 'text', 'text': msg['text']},]})
                    else:
                        message.append({'role': role, 'content': [{'type': 'text', 'text': msg},]})
        
        if isinstance(prompt, dict):
            message = []
            for role in ['system', 'user', 'assistant']:
                if role in prompt: 
                    msg = [{'role': role, 'content': []},]
                    if isinstance(prompt[role], str): 
                        msg['content'] = prompt[role]
                    else:
                        msg['content'].append({'type': 'text', 'text': prompt[role]['text']})
                        if "img_index" in prompt[role]:
                            img_index = prompt[role].pop('img_index')
                            base64_image = self.encode_image(image_path=image_paths[img_index])
                            msg['content'].append({'type': 'image_url', 'image_url': {"url": f"data:image/jepg;base64,{base64_image}", "detail": detail}})
                    message.append(msg)  
        # message += [
        #     {'role': 'user', 'content': [
        #         {'type': 'text', 'text': user_prompt},
        #         {'type': 'image_url', 'image_url': {"url": f"data:image/jepg;base64,{base64_image}", "detail": detail}}
        #     ]}
        # ]
        return message
    
    def get_response(self, image_path:str, prompt:Union[str, dict], temperature=1, max_new_tokens=1024, **kwargs: Any):
        response_str, response_state, history = self.chat(
            image_path, prompt, history=[], temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)

        return response_str, response_state
    
    def chat(self, image_path:str, prompt:Union[str, dict], history:list=[], temperature=1, max_new_tokens=1024, **kwargs: Any):
        message = history + self.format_query(image_path, prompt)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            # 'model': 'gpt-4-turbo',
            'model': 'gpt-4-vision-preview',
            'messages': message,
            'max_tokens': max_new_tokens,
            'seed': self.seed,
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response_state = response.status_code
        
        try: 
            response_json = response.json()
            response_str = response_json['choices'][0]['message']['content']
        except: 
            response_json, response_str = {}, ""

        if response_state == 200:
            history += message
            history += [{'role': 'assistant', 'content': response_str}]
        
        return response_str, response_state, history


class GeminiModel(BaseModel):

    def __init__(self, configs, **kwargs) -> None:
        super().__init__()
        import google.generativeai as genai

        genai.configure(api_key=configs['GEMINI_API_KEY'], transport='rest')
        self.model = genai.GenerativeModel(configs['GEMINI_MODEL'])
    
    def format_query(self, image_paths:str, prompt:Union[str, list, dict]):
        if isinstance(prompt, str):
            image = Image.open(image_paths)
            return [prompt, image]
        
        if isinstance(image_paths, str): image_paths = [image_paths]
        
        if isinstance(prompt, list):
            message = []
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image = Image.open(image_paths[img_index])
                    message.extend([msg['text'], image])
                else:
                    if isinstance(msg, dict): message.append(msg['text'])
                    else: message.append(msg)
            return message

    def get_response(self, image_path:str, prompt:str, temperature=1, max_new_tokens=1024, **kwargs: Any):
        """ 
        Return: 
        - response_str: (string) the decoded response
        - response_state: (int)
            STOP (1): Natural stop point of the model or provided stop sequence.
            MAX_TOKENS (2): The maximum number of tokens as specified in the request was reached.
            SAFETY (3): The candidate content was flagged for safety reasons.
            RECITATION (4): The candidate content was flagged for recitation reasons.
            OTHER (5): Unknown reason.
        """
        import google.generativeai as genai

        message = self.format_query(image_path, prompt)
        response = self.model.generate_content(message,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=temperature,
                max_output_tokens=max_new_tokens
            )
        )
        response_state = response.candidates[0].finish_reason

        try: 
            response_str = response.candidates[0].content.parts[0].text
            if response_state == 1: response_state = HTTPStatus.OK
        except: response_str = ""

        return response_str.strip(), response_state



qwenvl_special_str = """
Your response should be strictly structured in JSON format, consisting of the following keys and corresponding content:
{
    "THINK": <Analyze the logic behind your the next single-step action.>, 
    "NEXT": <Describe the next single-step action in words, e.g. "click on the .... located at ...".>,
    "ACTION": <Specify the precise API function name without arguments to be called to complete the user request, e.g., click_element. Leave it a empty string "" if you believe none of the API function is suitable for the task or the task is complete.>,
    "ARGS": <Specify the precise arguments in a dictionary format of the selected API function to be called to complete the user request, e.g., {"bbox": [0, 1, 2, 3]}. Leave it a empty dictionary {} if you the API does not require arguments, or you believe none of the API function is suitable for the task, or the task is complete.>
}
You should output only one next single-step action.

<YOUR RESPONSE>: 
"""

class QwenVLModel(BaseModel):

    def __init__(self, configs, **kwargs):
        super().__init__()
        from dashscope import MultiModalConversation

        self.api_key = configs['DASHSCOPE_API_KEY']
        self.api_version = configs['DASHSCOPE_MODEL']
        self.model = partial(
            MultiModalConversation.call, 
            model=self.api_version, api_key=self.api_key
        )
    
    def encode_image(self, image=None, image_path=None):
        assert image is not None or image_path is not None, \
            "Please specify at least an image array or a path to image."
        if image is not None:# convert image to bytes
            if isinstance(image, Image.Image): PIL_image = image
            elif isinstance(image, np.ndarray): PIL_image = Image.fromarray(image)
            else: raise TypeError(f"Unsupported image type {type(image)}")
            temp_directory = tempfile.gettempdir()
            unique_suffix = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
            filename = f"image{unique_suffix}.png"
            temp_image_path = os.path.join(temp_directory, filename)
            temp_image_url = f"file:///{temp_image_path}"
            temp_image_url = temp_image_url.replace('\\', '/')
            PIL_image.save(temp_image_path)
            image_url = temp_image_url
        elif image_path is not None: image_url = f'file://{os.path.abspath(image_path)}'
        else: raise NotImplementedError
        return image_url
    
    def format_query(self, image_paths:str, prompt:Union[str, dict], detail="high"):
        if isinstance(image_paths, str): image_paths = [image_paths]
        
        if isinstance(prompt, str):
            message = [{'role': 'user', 'content': [{'text': prompt},]}]
            if len(image_paths) != 0:
                for img_path in image_paths:
                    image_url = self.encode_image(image_path=img_path)
                    message[0]['content'].append({'image': image_url})
        
        if isinstance(prompt, list):
            message = []
            for role, msg in prompt:
                if isinstance(msg, dict) and "img_index" in msg:
                    img_index = msg.pop('img_index')
                    image_url = self.encode_image(image_path=image_paths[img_index])
                    message.append({'role': role, 'content': [
                        {'text': msg['text']}, {'image': image_url}]})
                else:
                    if isinstance(msg, dict):
                        message.append({'role': role, 'content': [{'text': msg['text']},]})
                    else:
                        message.append({'role': role, 'content': [{'text': msg},]})
                if role == "assistant": 
                    message.append({"role": "user", "content": [{"text": qwenvl_special_str}]})

        if isinstance(prompt, dict):
            message = []
            for role in ['system', 'user', 'assistant']:
                if role in prompt: 
                    msg = [{'role': role, 'content': []},]
                    if isinstance(prompt[role], str): 
                        msg['content'].append({'text': prompt[role]})
                    else:
                        msg['content'].append({'text': prompt[role]['text']})
                        if "img_index" in prompt[role]:
                            img_index = prompt[role].pop('img_index')
                            image_url = self.encode_image(image_path=image_paths[img_index])
                            msg['content'].append({'image': image_url})
                    message.append(msg)
        
        return message
    
    def chat(self, image_path:str, prompt:Union[str, dict], history:list=[], top_p=0.001, **kwargs: Any):
        message = history + self.format_query(image_path, prompt)

        response = self.model(messages=message, top_p=top_p)
        response_state = response.status_code

        try: 
            response_str = response.output.choices[0].message.content[0]['text']
            if response_state == HTTPStatus.OK:  #如果调用成功
                history += message
                history += [{
                    'role': 'assistant',
                    'content': [{'text': response_str}]
                }]
        except: response_str = ""

        return response_str.strip(), response_state, history

    def get_response(self, image_path:str, prompt:Union[str, dict], history:list=[], top_p=0.1, **kwargs: Any):
        response_str, response_state, history = self.chat(
            image_path, prompt, history=[], top_p=top_p, **kwargs)
        
        return response_str, response_state


def get_model(model_config, seed=2024) -> BaseModel:
    if model_config['NAME'].lower() == "openai": return OpenAIModel(model_config, seed)
    if model_config['NAME'].lower() == "gemini": return GeminiModel(model_config)
    if model_config['NAME'].lower() == "qwenvl": return QwenVLModel(model_config)
    raise NotImplementedError

