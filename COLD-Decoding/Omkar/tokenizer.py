# from transformers import GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import torch
import torch.nn.functional as F
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # print(tokenizer.get_vocab())

# word = tokenizer.convert_ids_to_tokens(14299)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"

print("Loading model")
model = GPT2LMHeadModel.from_pretrained(
    'gpt2-xl', output_hidden_states=True,
    resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)

model.to(device)
print("Loading Tokenizer")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

prompt = "The following is a sentence which contains the words: cat dance and money: "
x_ = tokenizer.encode(prompt, return_tensors='pt').to(device)

print(x_)

# Generate text based on the provided input
outputs = model(x_)

logits = outputs.logits[0, -1, :]

preds = F.softmax(outputs.logits, dim=-1)


tokenizer.batch_decode(sequences=preds)
# preds = torch.argsort(preds[0][0])

# print(preds[0:100])

#preds = preds


#x_final = [x.item() for x in list(torch.where(preds[0][0] > 0.0001)[0])]

# print(x_final)




# # Get the logits for the next token
# logits = output.logits[0, -1, :]

# probabilities = torch.softmax(logits, dim=-1)

# print(len(probabilities < 0.001))
