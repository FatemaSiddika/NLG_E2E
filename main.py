from dataloader import get_data_loader
from model import get_distilgpt2_model_and_tokenizer
from utils import train, evaluate

def main():
    batch_size = 16
    max_seq_len = 256
    epochs = 3
    learning_rate = 4e-5
    model, tokenizer = get_distilgpt2_model_and_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data_loader = get_data_loader(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size
    )

    print("Starting training...")
    train(model, data_loader, epochs, learning_rate)
    print("\nStarting evaluation...")
    evaluate(model, tokenizer, data_loader)

if __name__ == "__main__":
    main()
