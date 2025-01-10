import torch
import torch.nn as nn
from transformers import AutoTokenizer
from .configuration_evf import EvfConfig
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





class EvfSamModel(nn.Module):
    def __init__(self, config):
        super(EvfSamModel, self).__init__()
        self.config = config
        self.encoder_pretrained = 'BEIT3_weight/beit3_base_patch16_224.pth'  # 如果有预训练权重，可以在此指定路径
        self.initialize_evf_modules(config)

    def initialize_evf_modules(self, config):
        # 初始化 BEiT-3 模型
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(
                "model config should contain key 'mm_extractor_scale', with value 'base' or 'large'."
            )

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(beit_state_dict, strict=False)

        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        # 投影层
        in_dim = config.hidden_size
        assert in_dim == beit_config.encoder_embed_dim, (
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        )
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def forward(
            self,
            images_evf: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_masks: torch.BoolTensor,
            **kwargs,
    ):
        # 通过 BEiT-3 模型获取多模态特征
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )

        # 获取编码器输出并通过投影层
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)

        return feat


if __name__ == "__main__":
    # 定义配置
    config = EvfConfig()
    config.mm_extractor_scale = "base"  # 或者 "large"

    model = EvfSamModel(config)

    # 打印模型的参数量（转换为以百万为单位）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    print(f"参数总数：{total_params_m:.2f}M")
    print(f"可训练参数总数：{trainable_params_m:.2f}M")

    # 创建随机张量作为图像输入，尺寸为 (1, 3, 224, 224)
    images_evf = torch.randn(1, 3, 224, 224)

    # 准备文本输入 'defect'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer('defect', return_tensors='pt')
    input_ids = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask'].bool()

    # 前向传播
    features = model(
        images_evf=images_evf,
        input_ids=input_ids,
        attention_masks=attention_masks
    )

    # 打印输出形状
    print("输出形状:", features.shape)
