from monai.networks.nets import SwinUNETR, UNETR

from ..config import H, W
from .custom_nets import NNFormer2D, MedFormerUNet2D

def build_swinunetr():
    return SwinUNETR(
        img_size=(H, W),
        in_channels=5,
        out_channels=1,
        feature_size=48,
        spatial_dims=2,
    )

def build_unetr():
    return UNETR(
        img_size=(H, W),
        in_channels=5,
        out_channels=1,
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        spatial_dims=2,
    )

def build_nnformer():
    return NNFormer2D(in_ch=5, out_ch=1)

def build_medformer():
    return MedFormerUNet2D(in_channels=5, out_channels=1)

MODEL_BUILDERS = [
    ("SwinUNETR", build_swinunetr),
    ("UNETR", build_unetr),
    ("nnFormer", build_nnformer),
    ("MedFormer", build_medformer),
]