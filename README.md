# Volumetric Nearest Neighbor Upsampling in Torch

This repository contains a [Torch](http://torch.ch/) module for volumetric nearest
neighbor upsampling based on [kmul00/torch-vol](https://github.com/kmul00/torch-vol).
In the original repository, the nearest neighbor upsampling did not allow distinct
scaling factors for all three dimensions of the processed volumes; the module in this
repository changes that.

## Installation

Installation instructions roughly follow the
[instructions given in kmul00/torch-vol](https://github.com/kmul00/torch-vol/blob/master/INSTALL.md):

* Install Torch, for example using [torch/distro](https://github.com/torch/distro).
* Install the following requirements: [nnx](https://github.com/clementfarabet/lua---nnx)
  and [cunnx](https://github.com/nicholas-leonard/cunnx). Clone the repositories but
  _do not install them yet_. **Note that [torch/distro](https://github.com/torch/distro)
  might include `nnx` already.**
* Copy `VolumetricUpSamplingNearest.lua` from this repository into the `nnx` repository.
* Adapt `init.lua` (in the `nnx` repository) to include a line `require('nnx.VolumetricUpSamplingNearest')`.
* Copy `generic/VolumetricUpSamplingNearest.c` into `nnx/generic`.
* Adapt `init.c` (in `nnx`) to include the following lines:


    // Before function luaopen_libnnx
    #include "generic/VolumetricUpSamplingNearest.c"
    #include "THGenerateFloatTypes.h"
    // ...
    
    // In function luaopen_libnnx
    nn_FloatVolumetricUpSamplingNearest_init(L);
    nn_DoubleVolumetricUpSamplingNearest_init(L);

* Use `luarocks make nnx-0.1-1.rockspec` to build `nnx` including the volumetric
  upsampling module.
* After cloning `cunnx`, copy `cuda/VolumetricUpSamplingNearest.cu` to
  `cunnx`.
* Adapt `init.cu`:


    // Before luaopen_libcunnx.
    #include "VolumetricUpSamplingNearest.cu"
    // In luaopen_libcunnx.
    cunn_VolumetricUpSamplingNearest_init(L);
    // NOTE: cunn_ AND NOT cunnx_!

* Build `cunnx` using `luarocks make rocks/cunnx-scm-1.rockspec`.
* Run `th test.lua` to see if everything works correctly.

## Usage

Usage is very simple and illustrated in `test.lua`:

    local model = nn.Sequential()
    model:add(nn.VolumetricUpSamplingNearest(2, 3, 4))
    model:add(nn.VolumetricUpSamplingNearest(2, 3, 4))
    
    local input = torch.Tensor(8, 2, 10, 10, 10):fill(1)
    local goutput = torch.Tensor(8, 2, 40, 90, 160):fill(1)

    local output = model:forward(input)
    print(#output)
    local ginput = model:backward(input, goutput)
    print(#ginput)

Here, `VolumetricUpSamplingNearest` expects three arguments, the upsampling factors
in the first, second and third dimensions.

## License

**Original license:**

Copyright (c) [2015] [Koustav Mullick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**Changes:**

Copyright (c) 2017 David Stutz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.