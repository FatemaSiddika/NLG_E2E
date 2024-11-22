import torch
from tqdm import tqdm
from datasets import load_metric
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.cuda.amp as amp

def train(model, data_loader, epochs, learning_rate, warmup_steps=0):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = amp.GradScaler()
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(data_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            with amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(data_loader):.4f}")

    return model


def evaluate(model, tokenizer, data_loader):
    model.eval()
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")

    all_predictions = []
    all_references = []

    print("\nEvaluation Results:")
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            generated_ids = model.generate(input_ids, max_length=128, num_return_sequences=1)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(generated_text)
            all_references.extend([[ref] for ref in reference_text]) 

            for gen, ref in zip(generated_text[:2], reference_text[:2]): 
                print(f"Generated: {gen}")
                print(f"Reference: {ref}")

    # Calculate BLEU and ROUGE scores
    bleu_score = bleu.compute(predictions=all_predictions, references=all_references)["bleu"]
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)

    print("\nQuantitative Evaluation Results:")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"ROUGE Scores: {rouge_score}")


def print_hyperparameters(batch_size, max_seq_len, epochs, learning_rate):
    print("\nTraining Hyperparameters:")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
