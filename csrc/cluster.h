#pragma once

#include "extensions.h"

namespace cluster {
CLUSTER_API int64_t cuda_version() noexcept;

namespace detail {
CLUSTER_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace cluster

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, int64_t npoints, bool random_start);

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                     torch::Tensor ptr_y, double r, int64_t num_neighbors);
