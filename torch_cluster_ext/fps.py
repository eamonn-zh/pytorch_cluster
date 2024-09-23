import torch
from torch import Tensor


@torch.jit._overload  # noqa
def fps(src, npoints, ptr, random_start):  # noqa
    # type: (Tensor, int, Tensor, bool) -> Tensor  # noqa
    pass  # pragma: no cover


def fps(  # noqa
    src: Tensor,
    npoints: int,
    ptr: Tensor,
    random_start: bool = True,
):
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        src (Tensor): Point feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        npoints (int): Number of target points.
        ptr (Tensor): batch based on boundaries in CSR representation, *e.g.*,
            :obj:`ptr=[0,2,5,6]`.
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`LongTensor`

    """

    return torch.ops.torch_cluster_ext.fps(src, ptr, npoints, random_start)

