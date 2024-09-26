#pragma once

#include "../extensions.h"

void radius_cuda(torch::Tensor x, torch::Tensor y,
                 torch::optional<torch::Tensor> ptr_x,
                 torch::optional<torch::Tensor> ptr_y, torch::Tensor row, torch::Tensor col, torch::Tensor mask,
                 double r,
                 int64_t num_neighbors,
                 bool ignore_same_index);
