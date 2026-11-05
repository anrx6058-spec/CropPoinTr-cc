"""
模型注册中心
所有模型在这里注册，训练脚本通过名字自动加载
"""

from .pcn import PCN
from .rp_pcn import RP_PCN
from .jcs_net import JCS_Net

MODEL_REGISTRY = {
    'pcn': PCN,
    'rp_pcn': RP_PCN,
    'jcs_net': JCS_Net,
}

def get_model(name, **kwargs):
    """通过名字获取模型"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}. 可用模型: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

def list_models():
    """列出所有可用模型"""
    return list(MODEL_REGISTRY.keys())
