require 'torch'
require 'nn'
require 'optim'

values = torch.rand(3,4)-1
states = torch.eye(3)

output_size = 4
input_size = 3
hidden_units = 50

model = nn.Sequential()
model:add(nn.Linear(input_size, hidden_units))
model:add(nn.ReLU())
model:add(nn.Linear(hidden_units,output_size))
model:add(nn.ReLU())
model:training()
optimState = {
      learningRate = 0.01,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 0
   }

criterion = nn.MSECriterion()
parameters,gradParameters = model:getParameters()
optimMethod = optim.sgd

for i=1,5000 do
	
	local feval = function()
		local output = model:forward(states)
	    local err = criterion:forward(output, values)
		local df_do = criterion:backward(output, values)
	    model:backward(states, df_do)
	    gradParameters:div(3)
	    err = err/3
	    return err, gradParameters
	end
	optimMethod(feval, parameters, optimState)
end

model:evaluate()
local feval = function()
		local output = model:forward(states)
	    local err = criterion:forward(output, values)
	    return output,err/3
	end
output,err = feval(states)
print "Loss:"
print(err)
print "Expected:"
print(values)
print "Learned:"
print(output)

