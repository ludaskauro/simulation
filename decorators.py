import numpy as np

def entry(func):
    """
    Doesnt currently work correctly

    Args:
        func (_type_): _description_
    """
    def wrapper(self):
        if self.entryCondition:
            prevOutput = self.outputs
            prevInput = self.inputs
            prevMemory = self.memory
            #here comes one of the worst lines of code i have ever written
            entrySignal = list(list(self.entryCondition.values())[0].inputs.values())[0] 

            func(self)

            self.inputs = {key:value*entrySignal+prevInput[key]*(1-entrySignal) for key, value in self.inputs.items()}
            self.outputs = {key:value*entrySignal+prevOutput[key]*(1-entrySignal) for key, value in self.outputs.items()}
            self.memory = {key:value*entrySignal+prevMemory[key]*(1-entrySignal) for key, value in self.memory.items()}

        else:
            func(self)
        
        return None
    return wrapper

def to_single_feature(param, shape=(-1,1)):
    param = np.array(param)
    if len(param.shape) < 2:
        param = param.reshape(shape)
    return param

def vectorize(func):
    def wrapper(self):
        parameters = self.simulinkBlock.calibrationParameters.items()
        outputShape = [1]*len(parameters)
        for i,(name,param) in enumerate(parameters):
            param = to_single_feature(param.value)

            newShape = [1]*len(parameters)
            newShape[i] = -1

            param = param.reshape(newShape)

            self.simulinkBlock.calibrationParameters[name].value = param

            outputShape[i] = param.shape[i]
        
        self.outputShape = outputShape

        def reshape(value):
            return np.full(self.outputShape,value)
        
        for i,row in self.data.iterrows():
            row = row.map(reshape)
            func(self,row)

            yield self.simulinkBlock.outputs
            
    return wrapper