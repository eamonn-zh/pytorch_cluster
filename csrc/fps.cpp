#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>


#ifdef WITH_CUDA
#include "cuda/fps_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__fps_cuda(void) { return NULL; }

#endif
#endif
#endif

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, int64_t npoints, bool random_start) {
    return fps_cuda(src, ptr, npoints, random_start);
}

static auto registry = torch::RegisterOperators().op("torch_cluster_ext::fps", &fps);
