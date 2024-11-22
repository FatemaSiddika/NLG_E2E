from transformers import GPT2LMHeadModel, AutoTokenizer

def get_distilgpt2_model_and_tokenizer():
    
    model_name_or_path = "distilgpt2"  

    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer