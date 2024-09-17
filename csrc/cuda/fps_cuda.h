#pragma once

#include "../extensions.h"

torch::Tensor fps_cuda(torch::Tensor src, torch::Tensor ptr, int64_t npoints, bool random_start);
