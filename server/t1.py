# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="sardinelab/xTower13B", )#device_map="auto"
messages = [
    {
      "role": "user",
      "content": "apply_chat_template"
    },
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=1024, do_sample=False)
print(outputs)
