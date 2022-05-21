

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
