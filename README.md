# AgileIR
Implementation for "AgileIR: Memory-efficient Group Shifted Windows Attention for Agile Image Restoration"


# Acknowledgement
This work is supported in part through computational resources provided by the Data-Intensive Computing Centre, Universiti Malaya.  
The codebase is based on [KAIR](https://github.com/cszn/KAIR)

Cite us pls:
```
@inproceedings{10.1007/978-3-032-04546-1_30,
author = {Cai, Hongyi and Rahman, Mohammad Mahdinur and Wu, Jingyu and Dong, Wenzhen and Li, Jie},
title = {AgileIR: Memory-Efficient Group Shifted Windows Attention for&nbsp;Lightweight Image Restoration},
year = {2025},
isbn = {978-3-032-04545-4},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-032-04546-1_30},
doi = {10.1007/978-3-032-04546-1_30},
abstract = {Image Transformers show a magnificent success in Image Restoration tasks. Nevertheless, most of transformer-based models are strictly bounded by exorbitant memory occupancy. Our goal is to reduce the memory consumption of Swin Transformer and at the same time speed up the model during training process. Thus, we introduce AgileIR, group shifted attention mechanism along with window attention, which sparsely simplifies the model in architecture. We propose Group Shifted Window Attention (GSWA) to decompose Shift Window Multi-head Self Attention (SW-MSA) and Window Multi-head Self Attention (W-MSA) into groups across their attention heads, contributing to shrinking memory usage in back propagation. In addition to that, we keep shifted window masking and its shifted learnable biases during training, in order to induce the model interacting across windows within the channel. We also re-allocate projection parameters to accelerate attention matrix calculation, which we found a negligible decrease in performance. As a result of experiment, compared with our baseline SwinIR and other efficient quantization models, AgileIR keeps the performance still at 32.20 dB on Set5 evaluation dataset, exceeding other methods with tailor-made efficient methods and saves over 50\% memory while a large batch size is employed.},
booktitle = {Artificial Neural Networks and Machine Learning – ICANN 2025: 34th International Conference on Artificial Neural Networks, Kaunas, Lithuania, September 9–12, 2025, Proceedings, Part II},
pages = {364–375},
numpages = {12},
keywords = {Lightweight Super Resolution, Transformer, Memory Efficiency},
location = {Kaunas, Lithuania}
}
```
