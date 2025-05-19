import numpy as np

def entry(func):
    """
    Decorator for a global entry condition inside a simulink block. If there is an entry condition then inputs, outputs and memory will be unchanged
    if entry != 1
    """
    def wrapper(self):
        if self.entryCondition:
            #Store previous state
            prevOutput = self.outputs
            prevInput = self.inputs
            prevMemory = self.memory

            #here comes one of the worst line of code i have ever written
            entrySignal = list(list(self.entryCondition.values())[0].inputs.values())[0] 

            func(self) #compute output

            #determine whether to update state
            self.inputs = {key:value*entrySignal+prevInput[key]*(1-entrySignal) for key, value in self.inputs.items()}
            self.outputs = {key:value*entrySignal+prevOutput[key]*(1-entrySignal) for key, value in self.outputs.items()}
            self.memory = {key:value*entrySignal+prevMemory[key]*(1-entrySignal) for key, value in self.memory.items()}
            
        else:
            #if there is no entry condition then just compute the output
            func(self)
        
        return None
    return wrapper

def to_single_feature(param, shape=(-1,1)):
    """
    Convert param to numpy array and reshape

    Args:
        param (any): what to cast and reshape
        shape (tuple): reshape shape. Defaults to (-1,1).

    Returns:
        numpy.array: recast and reshaped array
    """
    param = np.array(param)
    if len(param.shape) < 2:
        param = param.reshape(shape)
    return param

def vectorize(computeOutput):
    """
    This wrapper vectorizes the simulation. This is done by checking the nr of calibration parameters and their shapes. 
    Then an output shape is determined which all input data is reshaped to before we run computeOutput.
    Returns a generator containing each time step of the simulation.

    Args:
        computeOutput (function): The function responsible for computing output in a simulink block
    """
    def wrapper(self):
        parameters = self.simulinkBlock.calibrationParameters.items()
        outputShape = [1]*len(parameters) #this is the shape which we will reshape our data to
        for i,(name,param) in enumerate(parameters):
            param = to_single_feature(param.value) #convert all parameters to numpy arrays and reshape them to a (-1,1) 

            newShape = [1]*len(parameters)
            newShape[i] = -1

            param = param.reshape(newShape) #Each parameter can hold the shape of a vector with elements in only one dimension
                                            #if there are 3 parameters and the third has 4 elements then that parameters new shape is (1,1,4)

            self.simulinkBlock.calibrationParameters[name].value = param #replace old parameter with reshaped one

            outputShape[i] = param.shape[i]
        
        self.outputShape = outputShape
        
        def reshape(value): #Simple reshape function which we can map our data to
            return np.full(self.outputShape,value)
        
        for i,row in self.data.iterrows(): #loop thorugh data
            row = row.map(reshape) #reshape data
            computeOutput(self,row) 

            yield self.simulinkBlock.outputs
            
    return wrapper