from transformers import (AutoConfig, 
                          PreTrainedTokenizer, 
                          AutoModelForMaskedLM,
                          AutoModelForCausalLM)
from typing import Union
import logging

logger = logging.getLogger(__name__)

def initialize_model(
    tokenizer: PreTrainedTokenizer, model_type: str, is_mlm: bool = True, **config
) -> Union[AutoModelForMaskedLM, AutoModelForCausalLM]:
    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(tokenizer.added_tokens_encoder),
        **config,
    )
    logger.info(f"Initializing model with config: {config}")
    auto_model = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM

    model = auto_model.from_config(config)

    return model