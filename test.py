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
    'hr' : os.listdir('../data/haerin_bench/haerin'),
    'non-hr': os.listdir('../data/haerin_bench/non-haerin')
}

prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: What is her name?
AI: ''']

images = [Image.open('../data/haerin_bench/haerin/' + photo_list['hr'][0])]


def test(): 
    inputs = processor(text = prompts, images =images,return_tensors='pt')
    inputs = {k:v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k : v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs,**generate_kwargs)
    sentence = tokenizere.decode(res.tolist()[0],skip_special_tokens=True)
    return sentence

print(test())


label_textt = 'Her Name is Haerin'
label = tokenizer(label_text,return_tensor='pt').to(device)
inputs = processor(text = prompts, images =images,return_tensors='pt')
inputs = {k:v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k : v.to(model.device) for k,v in inputs.items()}
print('input_ids :', inputs['input_ids'], inputs['input_ids'].size())
print('label :', label, label.size())

#Training only LM
for param in model.parameters():
    param.requires_grad = False

# vision_model 부분만 require_grad를 True로 설정
for param in model.language_model.parameters():
    param.requires_grad = True

mask_size = inputs['input_ids'][0].size()
non_padding_mask = torch.ones([1,mask_size-1]).to(device)
non_media_mask = torch.ones([1,mask_size-1]).to(device)
prompt_mask = torch.ones([1,mask_size-1]).to(device)

out = model.forward(**inputs,labels= label['input_ids'],num_images=torch.tensor([1]),non_padding_mask = non_padding_mask, non_media_mask = non_media_mask,prompt_mask = prompt_mask)
loss = out['loss']
optimizer = torch.optim.SGD(model.language_model.parameters(),lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(test())
#Training only Vision Model (Maybe Impossible)




#Training for Similarity Training
