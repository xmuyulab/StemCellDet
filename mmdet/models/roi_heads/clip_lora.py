import torch
import torch.nn as nn
import clip
from .class_name import ORGAN_DESC
import time
import torch.nn.functional as F

class clip_lora(nn.Module):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # load text feature
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
        self.clip_model.eval()
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.text_features_for_classes = []
        self.text_features_for_classes = torch.cat([self.clip_model.encode_text(clip.tokenize(desc).to(device)).detach() for desc in ORGAN_DESC], dim=0).float()
        self.text_features_for_classes = F.normalize(self.text_features_for_classes, p=2, dim=-1)

        self.lora_r = 8
        self.alpha = 0.8

        self.lora_embedding_A = nn.Parameter(torch.randn(self.clip_model.visual.proj.shape[0], self.lora_r),
                                             requires_grad=True)
        self.lora_embedding_B = nn.Parameter(torch.randn(self.lora_r, self.clip_model.visual.proj.shape[1]) * 0.0,
                                             requires_grad=True)

        nn.init.kaiming_uniform_(self.lora_embedding_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_embedding_B)

        # self._find_and_replace(self.clip_model)


    def forward(self, x):
        # x = self.clip_model.encode_image(x)
        x = x.type(self.clip_model.dtype)
        x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                        dtype=x.dtype, device=x.device),
                       x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.visual.ln_post(x[:, 0, :])

        x = (x @ self.clip_model.visual.proj) * (1 - self.alpha) + (x @ \
                                                                    self.lora_embedding_A.type(torch.float16) @ \
                                                                    self.lora_embedding_B.type(torch.float16)) * self.alpha
        return x

    # parent, target, target_name = _get_submodules(self.model, key)
    # new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
    # self._replace_module(parent, target_name, new_module, target)

    def _find_and_replace(self, model):
        for name, module in model.named_modules():
            if 'out_proj' in name:
                parent, target, target_name = _get_submodules(model, module, name)
                new_module = LoraLayer(target)
                self._replace_module(parent, target_name, new_module, target)


    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

def _get_submodules(model, module, name):
    name_list = [('[' + str(item) + ']') if item.isdigit() else item for item in name.split('.')]
    parent_name = ".".join(name_list[:-1]).replace('.[', '[')
    parent = eval('model.' + parent_name)
    target_name = name.split(".")[-1]
    target = module
    return parent, target, target_name

# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


import math
class LoraLayer(nn.Module):
    def __init__(
            self,
            proj,
            in_features: int = None,
            out_features: int = None,
            alpha: int = 0.8,
            lora_r: int = 8,
    ):
        super().__init__()
        if (in_features is None or out_features is None) and isinstance(proj, nn.Linear):
            in_features, out_features = proj.weight.shape
        # self.lora_embedding_A = nn.ParameterDict({'lora_embedding_A': nn.Parameter(torch.randn(in_features, r), requires_grad=True)})
        # self.lora_embedding_B = nn.ParameterDict({'lora_embedding_B': nn.Parameter(torch.randn(r, out_features)* 0.0, requires_grad=True)})
        self.lora_embedding_A = nn.Parameter(torch.randn(in_features, lora_r), requires_grad=True)
        self.lora_embedding_B = nn.Parameter(torch.randn(lora_r, out_features) * 0.0, requires_grad=True)
        # nn.ParameterDict({'lora_embedding_A': nn.Parameter(torch.randn(in_features, r), requires_grad=True)})
        # nn.ParameterDict({'lora_embedding_B': nn.Parameter(torch.randn(r, out_features)* 0.0, requires_grad=True)})

        self.in_features = in_features
        self.out_features = out_features
        self.proj = proj
        self.alpha = alpha
        self.lora_r = lora_r
        nn.init.kaiming_uniform_(self.lora_embedding_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_embedding_B)

    def forward(self, x):
        x = (x @ self.proj) * (1 - self.alpha) + (x @ self.lora_embedding_A @ x.lora_embedding_B) * self.alpha
        return x
