import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load both GPT-2 and Qwen models using HuggingFace `.from_pretrained` method.

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load GPT-2 (Completion Model) -----
gpt2_name = "gpt2"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device)
gpt2_model.eval()

# ----- Load Qwen (Instruction-Tuned Model) -----
# using a small but instruction-trained chat model
qwen_name = "Qwen/Qwen2-0.5B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_name).to(device)
qwen_model.eval()

print("Models loaded successfully:")
print(f"- Completion model     : {gpt2_name}")
print(f"- Instruction-tuned LLM: {qwen_name}")


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


tests = [("Once upon a time", "greedy"),
         ("What is 2+2?", "top_k"),
         ("Suggest a party theme.", "top_p")]

for prompt, strategy in tests:
    print(f"\n=== Prompt: {prompt!r} | Strategy: {strategy} ===")

    # GPT-2 output
    gpt2_out = generate(gpt2_model, gpt2_tokenizer, prompt, strategy=strategy, max_new_tokens=80)
    print("\n[GPT-2 Completion Model]")
    print(gpt2_out)

    # Qwen output
    qwen_out = generate(qwen_model, qwen_tokenizer, prompt, strategy=strategy, max_new_tokens=80)
    print("\n[Qwen Instruction-Tuned Model]")
    print(qwen_out)
