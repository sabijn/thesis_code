from transformers import (DataCollatorForLanguageModeling,
    AutoModelForMaskedLM, 
    AutoModelForCausalLM)


from tokenizer import create_tokenizer
from data import load_data
from model import initialize_model
from datasets import DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

import argparse
import logging

logger = logging.getLogger(__name__)

def initialize_trainer(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForLanguageModeling,
    datasets: DatasetDict,
    **config,
):
    args = TrainingArguments(**config)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
    )

    return trainer

def main():
    tokenizer = create_tokenizer('corpora/train_1.0.txt', min_freq=5)
    datasets = load_data(tokenizer, 'corpora')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    model = initialize_model(
        tokenizer, 
        'phueb/BabyBERTa-1', 
        num_hidden_layers=6, 
        intermediate_size=64,
        hidden_size=64,
        num_attention_heads=8,
    )

    print('#params', sum(param.numel() for param in model.parameters()))

    trainer = initialize_trainer(
        model, 
        tokenizer, 
        data_collator, 
        datasets, 
        output_dir='checkpoints', 
        save_steps=10_000, 
        eval_steps=100, 
        logging_steps=100,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=8,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        learning_rate=5e-4,
    )

    trainer.train()

if __name__ == '__main__':
    parser.add_argument()
    main()
