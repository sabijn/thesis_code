import os
from typing import Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast


def tokenize_wrapper(tokenizer):
    def tokenize(element, min_length=0, max_length=128):
        input_ids = [
            item
            for item in tokenizer(element["text"])["input_ids"]
            if max_length > len(item) > min_length
        ]
        return {"input_ids": input_ids}  #, "test": element["text"]}

    return tokenize


def load_data(
    args,
    tokenizer: PreTrainedTokenizerFast,
    data_dir: str,
    train_size: Optional[int] = None,
    dev_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> DatasetDict:

    raw_train = load_dataset("text", data_files=os.path.join(data_dir, f"train_sent_{args.version}_{args.top_k}.txt"))[
        "train"
    ]
    raw_dev = load_dataset("text", data_files=os.path.join(data_dir, f"dev_sent_{args.version}_{args.top_k}.txt"))[
        "train"
    ]
    raw_test = load_dataset("text", data_files=os.path.join(data_dir, f"test_sent_{args.version}_{args.top_k}.txt"))[
        "train"
    ]
    print(f'Generated datasets with the lengths of: {len(raw_train)} (train), {len(raw_dev)}, (dev), and {len(raw_test)} (test)')
    
    if train_size is not None:
        raw_train = raw_train.shuffle().select(range(train_size))
    if dev_size is not None:
        raw_dev = raw_dev.shuffle().select(range(dev_size))
    if test_size is not None:
        raw_test = raw_test.shuffle().select(range(test_size))

    raw_datasets = DatasetDict(
        {
            "train": raw_train,
            "eval": raw_dev,
            "test": raw_test
        }
    )

    tokenized_datasets = raw_datasets.map(
        tokenize_wrapper(tokenizer),
        batched=True,
    )

    return tokenized_datasets

def load_eval_data(
    args,
    tokenizer: PreTrainedTokenizerFast,
    data_file: str,
    test_size: Optional[int] = None,
) -> DatasetDict:
    
    # select the first 10k sentences from the test set
    raw_test = load_dataset("text", data_files=data_file)[
        "train"
    ]
    print(f'Generated datasets with the lengths of: {len(raw_test)} (test)')
    
    if test_size is not None:
        raw_test = raw_test.shuffle().select(range(test_size))

    raw_datasets = DatasetDict(
        {
            "test": raw_test
        }
    )

    tokenized_datasets = raw_datasets.map(
        tokenize_wrapper(tokenizer),
        batched=True,
    )

    return tokenized_datasets