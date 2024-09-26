#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

//#include "cpu/radius_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_cuda(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API void radius(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, torch::Tensor row, torch::Tensor col, torch::Tensor mask,
                     double r, int64_t num_neighbors,
                     bool ignore_same_index) {
    radius_cuda(x, y, ptr_x, ptr_y, row, col, mask, r, num_neighbors, ignore_same_index);
}

static auto registry = torch::RegisterOperators().op("torch_cluster_ext::radius", &radius);
