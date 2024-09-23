import torch

try:
    WITH_PTR_LIST = hasattr(torch.ops.torch_cluster_ext, 'fps_ptr_list')
except Exception:
    WITH_PTR_LIST = False
