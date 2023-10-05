# Load via Huggingface Style
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
import os

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

device= 'cuda:0'

model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
).to(device)

generate_kwargs = {
    'do_sample' : True,
    'top_k' : 5,
    'max_length' : 512,
}

photo_list = {
    'hr' : os.listdir('data/haerin_bench/haerin'),
    'non-hr': os.listdir('data/haerin_bench/non-haerin')
}


prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Who is she?
AI: ''']

images = [Image.open('data/haerin_bench/haerin/' + photo_list['hr'][0])]

inputs = processor(text = prompts, images =images,return_tensors='pt')
inputs = {k:v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k : v.to(model.device) for k,v in inputs.items()}
with torch.no_grad():
    res = model.generate(**inputs,**generate_kwargs)
sentence = tokenizere.decode(res.tolist()[0],skip_special_tokens=True)
print(sentence)
