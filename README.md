# LLM Projects

This repository contains hands-on projects to learn about Large Language Models (LLMs) from foundational concepts to practical implementations.

## Project Structure

### Project 1: LLM Playground ✅ DONE
**Location:** `llm_playground/`

The LLM Playground has been converted from Jupyter notebook to modular Python files for easier reusability and integration.

**What's Included:**
- ✅ Tokenization fundamentals (word-level, character-level, subword-level)
- ✅ Understanding BPE and TikToken
- ✅ Exploring Transformer architecture and GPT-2 internals
- ✅ Text generation with different decoding strategies (greedy, top-k, top-p)
- ✅ Comparing completion models vs. instruction-tuned models

**Structure:**
```
llm_playground/
├── tokenization/
│   ├── word_tokenizer.py       # Word-level tokenization
│   ├── char_tokenizer.py       # Character-level tokenization
│   ├── subword_tokenizer.py    # GPT-2 subword tokenization
│   └── tiktoken_comparison.py  # Compare GPT-2 and GPT-4 tokenizers
├── models/
│   ├── linear_layer.py              # Custom Linear layer implementation
│   ├── transformer_inspection.py    # Inspect GPT-2 transformer blocks
│   └── llm_output.py                # Predict next token from LLM
├── generation/
│   ├── text_generation.py      # Text generation with decoding strategies
│   └── model_comparison.py     # Compare GPT-2 vs Qwen models
└── main.py                     # Environment check and entry point
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- TikToken

### Installation

```bash
**Option 1: Run individual modules**
```bash
cd llm_playground/

# Run tokenization examples
python tokenization/word_tokenizer.py
python tokenization/char_tokenizer.py
python tokenization/tiktoken_comparison.py

# Run model examples
python models/linear_layer.py
python models/transformer_inspection.py
python models/llm_output.py

# Run generation examples
python generation/text_generation.py
python generation/model_comparison.py
```

**Option 2: Import functions in your own code**
```python
from tokenization.word_tokenizer import encode, decode
from generation.text_generation import generate
from generation.model_comparison import gpt2_model, qwen_model

# Use the functions
tokens = encode("Hello world")
print(decode(tokens))
```m text_generation import generate
from model_comparison import gpt2_model, qwen_model

# Use the functions
tokens = encode("Hello world")
print(decode(tokens))
```

**Option 3: Check environment**
```bash
python main.py
```

### What You'll Learn
- How text is converted into tokens that LLMs can process
- The architecture of Transformer-based models
- How to load and use pretrained models from Hugging Face
- Different text generation strategies and their trade-offs
- Key differences between base completion models and instruction-tuned models
