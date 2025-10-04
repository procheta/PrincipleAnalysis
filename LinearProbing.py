from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
# 1️⃣ Load Qwen model and tokenizer
model_name = "Qwen/Qwen-1_8B"   # change to your version, e.g. "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
model.to("cuda")
model.eval()  # freeze for feature extraction
# 2️⃣ Sample dataset
texts = [
    "The cat sat on the mat.",
    "Quantum computing will change the world.",
    "Football is a popular sport in Europe.",
    "Deep learning improves vision models."
]
texts=[]

def is_nan_value(x):
    if isinstance(x, str):
        return x.strip().lower() in ["nan", "na", "none", ""]
    return pd.isna(x) 

df=pd.read_csv("Principles/Principle1_sub.csv",sep="\t")
labels=[]
nan_rows = df[df['answer'].isna()]
for i in range(len(df)):
    x=df["answer"][i]
    try:
        if is_nan_value(float(x)):
            continue
    except:
        c=0
    x=df["input"][i]
   
    texts.append(str(x))
    x=df["answer"][i]
    x=x.strip()
    if x== "A":
        labels.append(0)
    else:
        labels.append(1)
print(len(labels))
print(len(df))
labels=torch.tensor(labels)
labels=labels.to("cuda")
tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

batch_size = 20
dataloader = DataLoader(texts[:400], batch_size=batch_size, shuffle=False)

train_len=int(len(texts)*0.80)
# 3️⃣ Tokenize inputs
train_label=labels[:400]
# 4️⃣ Get hidden states from all layers

# 5️⃣ Extract the 2nd layer embeddings (index 1 → 2nd layer)
# Each tensor: [batch_size, seq_len, hidden_dim]
# 6️⃣ Define a simple linear probe (e.g., binary classifier)
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


all_embeddings=[]
with torch.no_grad():
    for batch_texts in dataloader:
        # Tokenize this batch
        inputs = tokenizer(
            list(batch_texts),
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Move to GPU (if model is on cuda)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        # Get embeddings from 2nd layer, for example
        layer_2_embeds = hidden_states[1].mean(dim=1)
        layer_2_embeddings = layer_2_embeds.to(torch.float32).to("cuda")
        # Move to CPU and store for later use
        all_embeddings.append(layer_2_embeddings)



all_embeddings = torch.cat(all_embeddings, dim=0)
batch_size = 50
train_dataset = TensorDataset(all_embeddings, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

probe = LinearProbe(all_embeddings.size(-1), 2).to("cuda")

# 7️⃣ Train the probe
optimizer = optim.Adam(probe.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    total_loss=0
    for xb, yb in train_loader:
        xb = xb.to("cuda")
        yb = yb.to("cuda")
        logits = probe(xb)
        loss = criterion(logits,  yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader) 
    print(f"Epoch {epoch+1}: Loss = {avg_loss}")

# 8️⃣ Evaluate
test_label=labels[401:]



batch_size = 20
dataloader = DataLoader(texts[401:], batch_size=batch_size, shuffle=False)
all_embeddings=[]
print(len(dataloader))
with torch.no_grad():
    for batch_texts in dataloader:
        # Tokenize this batch
        inputs = tokenizer(
            list(batch_texts),
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Move to GPU (if model is on cuda)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        layer_2_embeddings = hidden_states[1].mean(dim=1)  # mean pooling over tokens
        layer_2_embeddings = layer_2_embeddings.to(torch.float32)
        all_embeddings.append(layer_2_embeddings)
        

all_embeddings = torch.cat(all_embeddings, dim=0)
batch_size = 20
test_dataset = TensorDataset(all_embeddings, test_label)
train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

total_acc=0
for xb, yb in train_loader:
        preds = torch.argmax(probe(xb), dim=1)
        acc = (preds == yb)
        accuracy = acc.float().mean().item()
        total_acc=total_acc+accuracy


print(total_acc/len(train_loader))

