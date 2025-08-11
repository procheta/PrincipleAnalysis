import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache



model = HookedTransformer.from_pretrained("qwen-1.8b", device="cuda:0")  # Correct name for pretrained model
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.to(model.cfg.device)

df=pd.read_csv("/users/psen/principles/Principle"+sys.argv[1]+".csv",sep="\t",on_bad_lines='skip')
ar=[]
for i in range(len(df)):
    text=df["Prompt"][i]
    try:
        output=model.generate(text,top_k=50,temperature=1)
        output=output.replace(text,"")
    except:
        continue
    if "A." or "B." in output:
        try:
            x=text+"###"+df["Response"][i]+ "###"+ df["Principle"][i]+"###"+output
            ar.append(x)
        except:
            c=0
    if(len(ar))>=100:
        break


with open("/users/psen/Output/Principle"+sys.argv[1]+"_Output.csv","w") as f:
    f.write("Prompt\tResponse\tPrinciple\tOutput\n")
    for text in ar:
        st=text.split("###")
        for i in range(len(st)):
            f.write(st[i])
            f.write("\t")
        f.write("\n")
f.close()
