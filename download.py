from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_model():
    model_name = "prithivida/parrot_paraphraser_on_T5"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save the tokenizer and model
    tokenizer.save_pretrained('./model')
    model.save_pretrained('./model')

if __name__ == "__main__":
    download_model()