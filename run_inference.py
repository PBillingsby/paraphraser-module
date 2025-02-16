import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained tokenizer and model from the specified directory
# The model is assumed to be stored locally at /model (pre-downloaded)
tokenizer = AutoTokenizer.from_pretrained("/model")
model = AutoModelForSeq2SeqLM.from_pretrained("/model")

# Retrieve the input text from the environment variable
# If INPUT_TEXT is not provided, it defaults to "Default input text"
INPUT_TEXT = os.getenv("input_text", "Default input text").strip()

# Tokenize the input text for model processing
# The prefix "paraphrase:" tells the model we want a paraphrased output
input_ids = tokenizer(
    f"paraphrase: {INPUT_TEXT}",  # Formatted input prompt
    return_tensors="pt",  # Return PyTorch tensors for model compatibility
    max_length=128,  # Limit the input length to prevent excessive memory usage
    truncation=True  # Ensures input is truncated if it exceeds max_length
).input_ids

# Generate multiple paraphrases using sampling-based text generation
outputs = model.generate(
    input_ids,
    num_return_sequences=3,  # Generate 3 paraphrased versions of the input
    max_length=128,  # Ensure the generated text does not exceed this length
    do_sample=True,  # Enable sampling for diverse outputs
    top_k=50,  # Consider the top 50 possible words at each step
    top_p=0.95,  # Nucleus sampling (keep most probable words, sum of probs ≤ 0.95)
    temperature=1.1  # Adjust randomness; higher = more creative
)

# Decode the generated outputs into human-readable text
paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Ensure the /outputs directory exists to store the generated results
os.makedirs("/outputs", exist_ok=True)

# Save the paraphrases to a JSON file
with open("/outputs/result.json", "w") as f:
    json.dump({"input_text": INPUT_TEXT, "paraphrases": paraphrases}, f, indent=2)
