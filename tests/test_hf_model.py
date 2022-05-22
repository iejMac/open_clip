import torch
from open_clip.hf_model import _POOLERS, PreTrainedTextEncoder
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
#test poolers
def test_poolers():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    mask = torch.ones(bs, sl, dtype=torch.long)
    mask[:2, 6:] = 0
    x = BaseModelOutput(h)
    for name, cls in _POOLERS.items():
        pooler = cls()
        res = pooler(x, mask)
        assert res.shape == (bs, d), f"{name} returned wrong shape"

#test PreTrainedTextENcoder
def test_pretrained_text_encoder():
    bs, sl, d = 2, 10, 64
    #TODO: run test for all supported archs here
    model_id = "arampacha/roberta-tiny"
    cfg = AutoConfig.from_pretrained(model_id)
    model = PreTrainedTextEncoder(model_id, d, proj='linear')

    x = torch.randint(0, cfg.vocab_size, (bs, sl))
    with torch.no_grad():
        emb = model(x)
    
    assert emb.shape == (bs, d) 
