from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class E2EDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading the E2E dataset from Hugging Face...")
        dataset = load_dataset("e2e_nlg", split="train") 
        self.samples = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample["meaning_representation"],
            max_length=self.max_seq_len,
            padding="max_length",  
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            sample["human_reference"],  
            max_length=self.max_seq_len,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

def get_data_loader(tokenizer, max_seq_len, batch_size):
    dataset = E2EDataset(tokenizer, max_seq_len)

    print(f"Total samples: {len(dataset)}")
    print(f"Tokenizer max sequence length: {max_seq_len}")
    print(f"Batch size: {batch_size}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
