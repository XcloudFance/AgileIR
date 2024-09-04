from models.network_swinir import SwinIR as swinir
from models.model_agileir import agileir 

def human_readable_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}P"
# 计算参数总量
model1 = agileir(img_size=512,patch_size=8,embed_dim=60, window_size=8, depths=[4,4,4,6])
model2 = swinir(img_size=512,patch_size=8,embed_dim=60, window_size=8)
total_params_1 = sum(p.numel() for p in model1.parameters())

total_params_2 = sum(p.numel() for p in model2.parameters())
print(f"agile模型的总参数量: {human_readable_number(total_params_1)}")
print(f"swin模型的总参数量: {human_readable_number(total_params_2)}")
# # 查看每层的参数量
# for name, param in model.named_parameters():
#     print(f"{name}: {human_readable_number(param.numel())}")