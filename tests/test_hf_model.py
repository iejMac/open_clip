import torch
from open_clip.hf_model import MeanPooler
from transformers.modeling_outputs import BaseModelOutput
#test poolers
def test_mean_pooler():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    print(h.mean(1))
    mask = torch.ones(bs, sl, dtype=torch.long)
    mask[:2, 6:] = 0
    x = BaseModelOutput(h)
    pooler = MeanPooler()
    res = pooler(x, mask)
    assert res.shape == (bs, d), "MeanPooler returned wrong shape"