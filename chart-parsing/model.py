import torch
import torch.nn as nn
from typing import Optional
from allennlp import SelfAttentiveSpanExtractor


class SpanProbe(nn.Module):
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0., **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.span_attention_extractor = SelfAttentiveSpanExtractor(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, unpooled, spans):
        span_embeddings = self.span_attention_extractor(unpooled, spans)
        span_embeddings = span_embeddings.view(-1, self.hidden_size)
        span_embeddings = self.dropout(span_embeddings)
        logits = self.classifier(span_embeddings)
        return logits


class Config:
    def __init__(self, **kwargs):
        for kwarg, val in kwargs.items():
            setattr(self, kwarg, val)

    def __repr__(self):
        representation = ""
        
        max_len = max(map(len, self.__dict__.keys()))
        for key, value in self.__dict__.items():
            str_value = str(value).split("\n")[0][:20]
            if len(str(value).split("\n")) > 1 or len(str(value).split("\n")[0]) > 20:
                str_value += " [..]"
            representation += (f"{key:<{max_len+3}}{str_value}\n")
            
        return representation

class ProbeConfig(Config):
    lr: float = 1e-2
    batch_size: int = 48
    epochs: int = 10
    verbose: bool = True
    num_items: Optional[int] = None
    weight_decay: float = 0.1


