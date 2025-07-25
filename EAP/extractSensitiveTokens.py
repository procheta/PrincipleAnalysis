import pandas as pd
import re
import pandas as pd
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig 
from transformer_lens.train import HookedTransformerTrainConfig, train
from tqdm import tqdm
from huggingface_hub import login





model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device="cuda:0" if torch.cuda.is_available() else "cpu")

df=pd.read_csv("Output/Principle12_Output.csv",sep="\t",index_col=False, on_bad_lines='skip')
count=1
with open("sensitive_tokens_principle12.csv","w") as f:
    f.write("prompt\tresponse\tprinciple\toutput\t sensitive\n")
    for i in range(len(df)):
        try:
            x= df["Prompt"][i]
            prompt="Given the following text "+x+ " Identify the specific words or phrases in the text that are most directly related to fulfilling the principle. List only the words/phrases, separated by commas."
            sensitive=model.generate(prompt,top_k=50,temperature=1)
            st=sensitive.split(",")
            for s in st:
                x=x.replace(s, "<mask>")
            f.write(df["Prompt"][i])
            f.write("\t")
            f.write(df["Response"][i])
            f.write("\t")
            f.write(df["Principle"][i])
            f.write("\t")
            f.write(str(df["Output"][i]))
            f.write("\t")
            f.write(x)
            f.write("\n")
            count=count+1
        except:
            c=0
            print("error")
        if count >=100:
            break
f.close()
