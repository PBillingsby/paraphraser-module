to build: docker build -t paraphraser:latest .

to run: docker run -e INPUT_TEXT="How can I rephrase this sentence?" -v $(pwd)/outputs:/outputs paraphraser:latest

AutoTokenizer
🔹 Purpose:

Converts raw text (string) into tokenized input that the model understands.
Also decodes model output back into human-readable text.
🔹 Example Workflow:

Encodes text (turns words into token IDs)
Decodes output (turns model predictions back into text)

2️⃣ AutoModelForSeq2SeqLM
🔹 Purpose:

Loads a pretrained sequence-to-sequence model (T5, BART, Pegasus, etc.).
Processes tokenized input and generates output tokens.
