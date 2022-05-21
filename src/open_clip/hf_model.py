""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import torch
import torch.nn as nn
from torch import TensorType
from traitlets import default
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig
except ImportError as e:
    transformers = None


# HF architecture dict:
arch_dict = {
  "roberta-base": {
    "config_names": {
      "context_length": "max_position_embeddings",
      "vocab_size": "vocab_size",
      "width": "hidden_size",
      "heads": "num_attention_heads",
      "layers": "num_hidden_layers",
    },
    "pooler": None, #TODO: find default roberta pooler
  }
}



# utils
# TODO: cls, max, mean, last
_POOLERS = {}

def register_pooler(cls):
    "Register pooler class"
    pass

class DummyPooler(nn.Module):
    "Fetches first of output hidden state"

    def __init__(self):
        pass
    
    def forward(self, x):
        return x.last_hidden_state[:, 0, :]

# arch-to-pooler mapping
_DEFAULT_POOLER = {}

def get_pooler(pooler_type:str):
    if pooler_type is None:
        # pooler_type = _DEFAULT_POOLER[self.config]
        pass
    return DummyPooler()

class PreTrainedTextEncoder(nn.Module):
    """HuggingFace model adapter
    
    # TODO: add dockstring here
    """
    def __init__(
            self, 
            model_name_or_path:str,
            embedding_dim:int,
            pooler_type:str=None,
            proj:str=None):
        super().__init__()
        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        
        self.pooler = get_pooler(pooler_type)
        d_model = getattr(self.config, arch_dict[model_name_or_path]["config_names"]["width"])
        if (d_model == embedding_dim) and (proj is None): # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == 'linear':
            self.proj == nn.Linear(d_model, embedding_dim, bias=False)
        elif proj == 'mlp':
            # TODO: add me
            pass

    def forward(self, x:TensorType) -> TensorType:
        # TODO: add attention mask?
        out = self.transformer(input_ids=x)
        pooled_out = self.pooler(out)

        return self.proj(pooled_out)

    def lock(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        # TODO: add support for partial freezing
        for n, p in self.transformer.named_parameters():
            if True: #mb optional LayerNorm params etc.
                p.requires_grad = False
