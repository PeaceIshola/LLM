import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "gpt2"
device = "cuda" if torch.cuda.is_available() else "mps"


# Step 1. Load GPT-2 model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# Step 2. Implement a text generation function using HuggingFace's generate method.
def generate(model, tokenizer, prompt, max_new_tokens=128):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)   # tokenize and move to device
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # Greedy decoding
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n GPT-2 | Greedy")
    print(generate(model, tokenizer, prompt, 80))


# Implement `generate` to support 3 strategies: greedy, top_k, and top_o
# You may find this link helpful: https://huggingface.co/docs/transformers/en/main_classes/text_generation

def generate(model, tokenizer, prompt, strategy="greedy", max_new_tokens=128):

    # Tokenize prompt and move to model's device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Configure decoding based on chosen strategy
    if strategy == "greedy":
        # Always select highest-probability token
        gen_kwargs = dict(do_sample=False)

    elif strategy == "top_k":
        # Sample from top-k most likely tokens
        gen_kwargs = dict(
            do_sample=True,
            top_k=50,            # You may tune this (10-100 typical)
            top_p=1.0,           # Disabled for top-k
            temperature=1.0
        )

    elif strategy == "top_p":
        # Nucleus sampling: sample from tokens whose cumulative prob ≥ top_p
        gen_kwargs = dict(
            do_sample=True,
            top_k=0,             # Disabled for top-p (or leave unrestricted)
            top_p=0.9,           # You may tune this (0.8-0.95 typical)
            temperature=1.0
        )

    else:
        raise ValueError("strategy must be one of: 'greedy', 'top_k', or 'top_p'")

    # Generate output token IDs
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )

    # Decode token IDs into string
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


tests = ["Once upon a time", "What is 2+2?", "Suggest a party theme."]

for strategy in ["greedy", "top_k", "top_p"]:
    print(f"\n===== Strategy: {strategy.upper()} =====")
    for prompt in tests:
        print(f"\nPrompt: {prompt}")
        print(generate(model, tokenizer, prompt, strategy=strategy, max_new_tokens=60))
