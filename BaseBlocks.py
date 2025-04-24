from BaseClasses import Block
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import plotly.graph_objects as go
from decorators import entry
from functools import reduce

class EntryCondition(Block):
    def __init__(self, name: str, entrySignal) -> None:
        super().__init__(name, [entrySignal], [])        

        self.label = 'EntryCond'

        self.entrySignal = entrySignal
    
    def computeOutput(self):
        pass


class SampleTime(Block):
    def __init__(self, name: str) -> None:
        super().__init__(name, [], [name])

        self.label = 'ts'

        self.ready = {i:True for i in self.ready.keys()}
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.raster
    
    def reset(self):
        self.visited = False

class Constant(Block):
    def __init__(self, name: str, value) -> None:
        super().__init__(name, [], [name])

        self.label = str(value)

        self.ready = {i:True for i in self.ready.keys()}

        self.value = value
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.value
    
    def reset(self):
        self.visited = False

class Parameter(Block):
    def __init__(self, name, value=None) -> None:
        super().__init__(name, [], [name])
        
        self.label = name

        self.ready = {i:True for i in self.ready.keys()}

        self.value = value

        self.calibrationParameters[name] = value
    @entry
    def computeOutput(self):
        self.outputs[self.name] = self.value
    
    def reset(self):
        self.visited = False

class Delay(Block):
    def __init__(self, name, signalIn, signalOut) -> None:
        super().__init__(name, [signalIn], [signalOut])

        self.label = '1/z'

        self.memory = {i:0 for i in [signalOut]}
        self.ready = {i:True for i in [signalIn]}

        self.signalOut = signalOut
        self.signalIn = signalIn

    @entry
    def computeOutput(self):
        self.outputs = self.memory
        self.memory[self.signalOut] = self.inputs[self.signalIn]
    
    def reset(self):
        self.visited = False
        
class Add(Block):
    def __init__(self, name, pos1,pos2, output) -> None:
        super().__init__(name,[pos1,pos2],[output])
        
        self.label = 'Add'

        self.inputPorts = {pos1:'+',pos2:'+'}
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] + self.inputs[self.inputs_list[1]]

class Subtract(Block):
    def __init__(self, name, pos, neg, output) -> None:
        super().__init__(name,[pos,neg],[output])
        
        self.label = 'Subtract'

        self.inputPorts = {pos:'+',neg:'-'}
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] - self.inputs[self.inputs_list[1]]

class Min(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name,[signal1,signal2],[output])
        
        self.label = 'Min'
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.minimum(self.inputs[self.inputs_list[0]], self.inputs[self.inputs_list[1]])

class Max(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name,[signal1,signal2],[output])
        
        self.label = 'Max'
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.maximum(self.inputs[self.inputs_list[0]], self.inputs[self.inputs_list[1]])
    

class Multiply(Block):
    def __init__(self, name, factor1, factor2, output) -> None:
        super().__init__(name,[factor1,factor2],[output])
        
        self.label = 'Multiply'

        self.inputPorts = {factor1:'x',factor2:'x'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]]

class Divide(Block):
    def __init__(self, name, numerator, denominator, output) -> None:
        super().__init__(name, [numerator,denominator], [output])
        
        self.label = 'Divide'

        self.inputPorts = {numerator:'x', denominator:'/'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] / (self.inputs[self.inputs_list[1]]+0.01)

class And(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name, [signal1,signal2], [output])

        self.label = 'And'
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]]

class Not(Block):
    def __init__(self, name, signal, output) -> None:
        super().__init__(name, [signal], [output])

        self.label = 'Not'
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = 1-self.inputs[self.inputs_list[0]]

class Or(Block):
    def __init__(self, name, signal1,signal2, output) -> None:
        super().__init__(name, [signal1,signal2], [output])

        self.label = 'Or'
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] + self.inputs[self.inputs_list[1]] - (self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]])

class Switch(Block):
    def __init__(self, name, switched, default, condition, output) -> None:
        super().__init__(name, [switched,default,condition], [output])

        self.label = 'Switch'

        self.inputPorts = {switched:'Switched', default:'Default', condition:'Condition'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[1]]*(1-self.inputs[self.inputs_list[2]]) + self.inputs[self.inputs_list[0]]*self.inputs[self.inputs_list[2]]

class Abs(Block):
    def __init__(self, name, signal, output) -> None:
        super().__init__(name, [signal], [output])

        self.label = '|u|'
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.abs(self.inputs[self.inputs_list[0]])

class GreaterThanOrEqual(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '>='

        self.inputPorts = {first:'First', second:'Second'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] >= self.inputs[self.inputs_list[1]]

class GreaterThan(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '>'
        
        self.inputPorts = {first:'First', second:'Second'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] > self.inputs[self.inputs_list[1]]

class LessThanOrEqual(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '<='

        self.inputPorts = {first:'First', second:'Second'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] <= self.inputs[self.inputs_list[1]]

class LessThan(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '<'

        self.inputPorts = {first:'First', second:'Second'}
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] < self.inputs[self.inputs_list[1]]

class EqualTo(Block):
    def __init__(self, name, first, second, output):
        super().__init__(name, [first,second], [output])
        self.label = '=='        
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] == self.inputs[self.inputs_list[1]]

class SampleDelay(Block):
    def __init__(self, name, signal, delay, output):
        super().__init__(name, [signal, delay], [output])

        self.label = 'Sample Delay'

        self.inputPorts = {signal:'In',delay:'td'}
        
        self.memory = []
    @entry
    def computeOutput(self):
        if len(self.memory) == self.inputs[self.inputs_list[1]]:
            self.outputs[self.outputs_list[0]] = self.memory.pop(0)
        
        self.memory.append(self.inputs[self.inputs_list[0]])

class FlankUp(Block):
    def __init__(self, name, input, output):
        super().__init__(name, [input], [output])
        
        self.label = 'Flank Up'

        self.memory = {'prevSignal':0, 'prevOutput':0}
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = (1-self.memory['prevSignal'])*(1-self.memory['prevOutput'])*self.inputs[self.inputs_list[0]]
        
        self.memory['prevOutput'] = self.outputs[self.outputs_list[0]]
        self.memory['prevSignal'] = self.inputs[self.inputs_list[0]]

class FlankDown(Block):
    def __init__(self, name, input, output):
        super().__init__(name, [input], [output])
        
        self.label = 'Flank Down'

        self.memory = {'prevSignal':0, 'prevOutput':0}
    
    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.memory['prevSignal']*(1-self.memory['prevOutput'])*(1-self.inputs[self.inputs_list[0]])
        
        self.memory['prevOutput'] = self.outputs[self.outputs_list[0]]
        self.memory['prevSignal'] = self.inputs[self.inputs_list[0]]

class Input(Block):
    def __init__(self, name, outputs) -> None:
        super().__init__(name,outputs,outputs)
        self.label = 'Input'
    @entry
    def computeOutput(self):
        self.outputs = self.inputs

class Output(Block):
    def __init__(self, name, inputs) -> None:
        super().__init__(name,inputs,inputs)
        
        self.label = 'Output'
    @entry
    def computeOutput(self):
        self.outputs = self.inputs

class Map2D(Block):
    def __init__(self,name,x,y,output,DCMPath):
        super().__init__(name, [x,y], [output])
        
        self.label = '2D Map ' + name.split('_')[-1]

        self.inputPorts = {x:'x',y:'y'}

        self.outputPorts = {output:'z'}
    
        with open(DCMPath,'r') as file:
            map_started = False
            x,y,z = [],[],[]
            
            for line in file.readlines():
                line = line.strip()

                if line.startswith('WERT'):
                    tmp = line.split()[1:]
                    z.append([float(i) for i in tmp])
                
                if line.startswith('*SSTX') or line.startswith('*SSTY'):
                    map_started = True
                
                if line.startswith('END') and map_started:
                    break

                if line.startswith('ST/X'):
                    tmp = line.split()[1:]
                    x.extend([float(i) for i in tmp])
                
                if line.startswith('ST/Y'):
                    y.append(float(line.split()[1]))
        
        self.xPoint = x
        self.yPoint = y

        self.grid = RegularGridInterpolator(points=(x,y),values=z,bounds_error=False)
    @entry
    def computeOutput(self):        
        self.inputs = {signal:np.clip(value,min(coord),max(coord)) for coord, (signal, value) in zip([self.yPoint,self.xPoint], self.inputs.items())}
        self.outputs[self.outputs_list[0]] = self.grid((self.inputs[self.inputs_list[1]],self.inputs[self.inputs_list[0]]))

    def visualizeMap(self):
        x, y = np.meshgrid(self.yPoint, self.xPoint)
        z = self.grid.values

        lines = []
        line_marker = dict(color='#000000', width=4)
        for i, j, k in zip(x, y, z):
            lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
        for i, j, k in zip(x.T, y.T, z.T):
            lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))


        trace2 = go.Surface(z=z, x=x, y=y,showscale=False)

        fig = go.Figure(lines)
        fig.add_trace(trace2)

        fig.update_layout(
            title=self.name,
            scene=dict(
                xaxis_title=self.inputs_list[0],
                yaxis_title=self.inputs_list[1],
                zaxis_title=self.outputs_list[0]
            ), 
            showlegend=False
        )    
        fig.show()

class Map1D(Block):
    def __init__(self,name,x,output,DCMPath):
        super().__init__(name, [x], [output])
        
        self.label = '1D Map ' + name.split('_')[-1]

        self.inputPorts = {x:'x'}
        self.outputPorts = {output:'y'}
    
        with open(DCMPath,'r') as file:
            map_started = False
            x,y = [],[]
            
            for line in file.readlines():
                line = line.strip()

                if line.startswith('WERT'):
                    tmp = line.split()[1:]
                    y.extend([float(i) for i in tmp])
                
                if line.startswith('*SSTX') or line.startswith('*SSTY'):
                    map_started = True
                
                if line.startswith('END') and map_started:
                    break

                if line.startswith('ST/X'):
                    tmp = line.split()[1:]
                    x.extend([float(i) for i in tmp])
                
        self.xPoint = x

        self.grid = RegularGridInterpolator(points=(x,),values=y,bounds_error=False)
    @entry
    def computeOutput(self):  
        #not super pretty      
        self.inputs = {signal:np.clip(value,min(self.xPoint),max(self.xPoint)) for signal, value in self.inputs.items()}
        data = self.inputs[self.inputs_list[0]]
        data_shape = data.shape
        data = data.flatten()
        func = lambda x:self.grid([x])
        self.outputs[self.outputs_list[0]] = np.array(list(map(func,data))).reshape(data_shape)

    def visualizeMap(self):
        fig = go.Figure(data=go.Scatter(x=self.xPoint, y=self.grid.values))
        fig.update_layout(
            title=self.name,
            xaxis_title=self.inputs_list[0],
            yaxis_title=self.outputs_list[0],
            showlegend=False
        )    
        fig.show()

class BitWiseAnd(Block):
    def __init__(self, name, bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7, output):
        super().__init__(name, [bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7], [output])
        
        self.label = 'Bitwise And'

        self.inputPorts = { bit0:'bit0', bit1:'bit1', bit2:'bit2', bit3:'bit3', bit4:'bit4', bit5:'bit5', bit6:'bit6', bit7:'bit7'}

    @entry
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = reduce(lambda x,y: x*y, self.inputs.values())

class Clip(Block):
    def __init__(self, name, input, max, min, output):
        super().__init__(name, [input, max, min], [output])

        self.label = 'clamp'

    def computeOutput(self):
        signal = self.inputs[self.inputs_list[0]]
        maxVal = self.inputs[self.inputs_list[1]]
        minVal = self.inputs[self.inputs_list[2]]

        self.outputs[self.outputs_list[0]] = np.clip(a=signal, a_max=maxVal, a_min=minVal)

class NotEqual(Block):
    def __init__(self, name, signal1, signal2, output):
        super().__init__(name, [signal1, signal2], [output])
        
        self.label = '!='

    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] != self.inputs[self.inputs_list[1]]