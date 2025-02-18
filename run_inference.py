import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model from local directory (/model)
tokenizer = AutoTokenizer.from_pretrained("/model")
model = AutoModelForSeq2SeqLM.from_pretrained("/model")

# Retrieve input text from environment variable (defaults to "Default input text" if not set)
INPUT_TEXT = os.getenv("input_text", "Default input text").strip()

# Tokenize input text for the model
input_ids = tokenizer(
    f"paraphrase: {INPUT_TEXT}",  # Adds "paraphrase:" prefix to indicate task
    return_tensors="pt",  # Returns PyTorch tensors (needed for model processing)
    max_length=128,  # Limits tokenized input length to prevent excessive memory usage
    truncation=True  # Truncate if input exceeds max_length
).input_ids

# Generate paraphrased text using sampling-based text generation
outputs = model.generate(
    input_ids,  # Tokenized input for the model
    num_return_sequences=3,  # Generate 3 different paraphrased outputs
    max_length=128,  # Max length for generated text (prevents overflow)
    do_sample=True,  # Enables non-deterministic output generation
    top_k=50,  # Considers only the top 50 word choices at each step
    top_p=0.95,  # Nucleus sampling: retains words until cumulative probability ≥ 0.95
    temperature=1.1  # Controls randomness: higher values = more diverse output
)

# Decode generated outputs into readable text
paraphrases = [
    tokenizer.decode(output, skip_special_tokens=True) for output in outputs
]

# Ensure output directory exists
os.makedirs("/outputs", exist_ok=True)

# Save input and paraphrased results to a JSON file
with open("/outputs/result.json", "w") as f:
    json.dump({"input_text": INPUT_TEXT, "paraphrases": paraphrases}, f, indent=2)
