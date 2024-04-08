from transformers import AutoConfig, PreTrainedTokenizer, AutoModelForMaskedLM


def initialize_model(
    tokenizer: PreTrainedTokenizer, model_type: str, **config
) -> AutoModelForMaskedLM:
    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(tokenizer.added_tokens_encoder),
        **config,
    )

    model = AutoModelForMaskedLM.from_config(config)

    return model