local VolumetricUpSamplingNearest, parent = torch.class('nn.VolumetricUpSamplingNearest', 'nn.Module')

function VolumetricUpSamplingNearest:__init(scale_t, scale_y, scale_x)
   parent.__init(self)

   self.scale_factor_t = scale_t
   self.scale_factor_y = scale_y
   self.scale_factor_x = scale_x
   
   if self.scale_factor_t < 1 or self.scale_factor_y < 1 or self.scale_factor_x < 1 then
     error('scale_factor must be greater than 1')
   end
   if math.floor(self.scale_factor_t) ~= self.scale_factor_t or math.floor(self.scale_factor_y) ~= self.scale_factor_y or math.floor(self.scale_factor_x) ~= self.scale_factor_x then
     error('scale_factor must be integer')
   end
   self.inputSize = torch.LongStorage(5)
   self.outputSize = torch.LongStorage(5)
   self.usage = nil
end

function VolumetricUpSamplingNearest:updateOutput(input)
   if input:dim() ~= 5 and input:dim() ~= 4 then
     error('VolumetricUpSamplingNearest only support 4D or 5D tensors')
   end
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   local tdim = input:dim() - 2
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[tdim] = self.outputSize[tdim] * self.scale_factor_t
   self.outputSize[ydim] = self.outputSize[ydim] * self.scale_factor_y
   self.outputSize[xdim] = self.outputSize[xdim] * self.scale_factor_x
   -- Resize the output if needed
   if input:dim() == 4 then
     self.output:resize(self.outputSize[1], self.outputSize[2],
       self.outputSize[3],self.outputSize[4])
   else
     self.output:resize(self.outputSize)
   end
   input.nn.VolumetricUpSamplingNearest_updateOutput(self, input)
   return self.output
end

function VolumetricUpSamplingNearest:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.VolumetricUpSamplingNearest_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function VolumetricUpSamplingNearest:__tostring__()
   return string.format('%s(%d, %d, %d)', torch.type(self),
          self.scale_factor_t, self.scale_factor_y, self.scale_factor_x)
end