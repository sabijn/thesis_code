from transformers import (DataCollatorForLanguageModeling,
    AutoModelForMaskedLM, 
    AutoModelForCausalLM)


from tokenizer import create_tokenizer
from data import load_data
from model import initialize_model
from argparser import create_arg_parser

from datasets import DatasetDict
from typing import Union
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

import logging

logger = logging.getLogger(__name__)

def initialize_trainer(
    model: Union[AutoModelForMaskedLM, AutoModelForCausalLM],
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

def main(args):
    """
    Training model
    """
    tokenizer = create_tokenizer(f'{args.data_dir}/train_sent_{args.version}_{args.top_k}.txt', min_freq=5)
    datasets = load_data(args, tokenizer, args.data_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    if args.base_model == 'phueb/BabyBERTa-1':
        model = initialize_model(
            tokenizer, 
            args.base_model, 
            is_mlm=True,
            num_hidden_layers=6, 
            intermediate_size=64,
            hidden_size=64,
            num_attention_heads=8,
        )
    elif args.base_model == 'distilgpt2':
        model = initialize_model(
            tokenizer, 
            args.base_model, #'microsoft/deberta-v3-base',  # 'phueb/BabyBERTa-1', 
            num_hidden_layers=8, 
            intermediate_size=256,
            hidden_size=256,
            num_attention_heads=8,
            is_mlm=False,
        )
    else:
        logger.critical('Model not implemented')
        raise NotImplementedError

    print('#params', sum(param.numel() for param in model.parameters()))

    trainer = initialize_trainer(
        model, 
        tokenizer, 
        data_collator, 
        datasets, 
        output_dir=args.output_dir, 
        save_steps=args.save_steps, 
        eval_steps=args.eval_steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        group_by_length=args.group_by_length,
        auto_find_batch_size=args.auto_find_batch_size,
        do_eval=args.do_eval,
        evaluation_strategy=args.evaluation_strategy
    )

    trainer.train()
    trainer._save_checkpoint(trainer.model, None)

if __name__ == '__main__':
    args = create_arg_parser()
    main(args)
