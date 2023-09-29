# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


import re
import copy

# %%
def preprocess_text(text) -> str:
    text_copy = copy.copy(text)
    text_copy = re.sub(r'[^\w\s]', '', text_copy)
    text_copy = re.sub(r'\s+', ' ', text_copy) 
    text_copy = re.sub(r'\n', ' ', text_copy)
    text_copy = re.sub(r'\d', '', text_copy)
    text_copy = text_copy.lower()
    return text_copy

# %%
def encoding_maps(chars) -> tuple:
    char_to_idx = {char:idx+1 for idx, char in enumerate(chars)}
    char_to_idx['.'] = 0
    idx_to_char = {idx:char for char, idx in char_to_idx.items()}

    return (char_to_idx, idx_to_char)

# %%
def create_dataset(words: str, block_size = 3, char_to_idx_map: dict = None) -> dict:
    
    xs, ys = [], []

    #make efficient with pytorch
    for word in words:
        #print(word)
        context = [0] * block_size
        
        for ch in word + ".":
            idx = char_to_idx_map[ch]
            xs.append(context)
            ys.append(idx)
            
            #print(''.join(idx_to_char[i] for i in context), '--->', idx_to_char[idx])
            context = context[1:] + [idx]

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return {"xs": xs, "ys": ys}

# %%
def init(param_size_dict: dict, seed = 2147483647) -> dict:
    g = torch.Generator().manual_seed(seed)
    
    params = {}
    for key, param_size in param_size_dict.items():
        params[key] = torch.randn(param_size, generator = g)
        
    print("Parameters initalized: ")
    for param_name, data in params.items:
        print(f"Param: {param_name}, shape: {data.shape}")
    return params

# %%



