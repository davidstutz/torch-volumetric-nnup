require('torch')
require('cutorch')
require('nn')
require('cunn')
-- Volumetric upsampling is integrated in nnx and cunnx!
require('nnx')
require('cunnx')

local model = nn.Sequential()
model:add(nn.VolumetricUpSamplingNearest(2, 3, 4))
model:add(nn.VolumetricUpSamplingNearest(2, 3, 4))

local input = torch.Tensor(8, 2, 10, 10, 10):fill(1)
local goutput = torch.Tensor(8, 2, 40, 90, 160):fill(1)
local output = model:forward(input)
print(#output)
local ginput = model:backward(input, goutput)
print(#ginput)

model = model:cuda()
input = input:cuda()
goutput = goutput:cuda()
output = model:forward(input)
print(#output)
ginput = model:backward(input, goutput)
print(#ginput)



