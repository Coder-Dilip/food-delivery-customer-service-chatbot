import json
from channels.generic.websocket import AsyncWebsocketConsumer

import torch
# from home.models import load_model_and_tokenizer
from delivery_support.settings import BASE_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from channels.layers import get_channel_layer
import google.generativeai as genai
from decouple import config

# Configure the SDK with your API key
genai.configure(api_key=config('GOOGLE_API_KEY'))
def load_model_and_tokenizer(model_path, base_dir=None):
    """Loads the LLM model and tokenizer from the specified path."""

    # Use base_dir if provided, otherwise assume current directory structure
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(base_dir, 'llama-2-custom')

    # Load quantization configuration
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


# Load the model and tokenizer during module import (optional)
model, tokenizer = load_model_and_tokenizer(
    os.path.join(BASE_DIR, 'delivery_support', 'llama-2-custom')
)
import asyncio  # Import asyncio for asynchronous delay

from transformers import pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread

class RecipeConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.channel_layer = get_channel_layer()
        # await self.channel_layer.group_add("generation_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        # await self.channel_layer.group_discard("generation_group", self.channel_name)
        pass




    async def generate_text(self, prompt, num_new_tokens):
        # Count the number of tokens in the prompt
        num_prompt_tokens = len(tokenizer(prompt)['input_ids'])

        # Calculate the maximum length for the generation
        max_length = num_prompt_tokens + num_new_tokens

        # Initialize the pipeline with the streamer
        inputs = tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer)

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        async def send_text_to_frontend(text):
            await self.send(text_data=json.dumps({'text': text}))

        async def generate_and_send():
            generated_text = ""
        
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end=" ")  # Output to terminal for debugging
                await send_text_to_frontend(new_text)  # Send text to frontend in real-time
                await asyncio.sleep(0.01)  # Add a small delay to control the rate of sending

        # Start generating and sending text in the background
        asyncio.create_task(generate_and_send())






    async def handle_text_generation(self, event):
        message = event['text']
        num_new_tokens=300
        generated_text = await self.generate_text(message, num_new_tokens)
        for chunk in generated_text:
            await self.send(text_data=json.dumps({'text': chunk}))

    async def receive(self, text_data):
        data = json.loads(text_data)
        query = data.get('query')
        prev_question = data.get('prev_question')
        prev_answer = data.get('prev_answer')

       
     

        

        if len(prev_question)==0:
          
            system_message = 'You are Food Fast(food delivery company chatbot), Assume yourself as a customer support for a food delivery company, you are going to get questions related to payment methods or canceling the food order or question related to food and price by the users from Nepal, answer accordingly with shortly and accurately'
        else:
          
            system_message = 'You do not need to greet or say hello, just reply directly. Assume yourself as a customer support for a food delivery company, you are going to get questions related to payment methods or canceling the food order or question related to food and price by the users from Nepal, answer accordingly with shortly and accurately' + " " + "Have a look into the previous chat before answering, if the question is related to previous response then only look to this previous conversation for answering to new question ==>" + f" question '{prev_question} and answer {prev_answer}'"

       

        prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{query} [/INST]"
        num_new_tokens = 300
        await self.generate_text(prompt,num_new_tokens=300)
   


