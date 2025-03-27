from BaseClasses import Block
from scipy.interpolate import RegularGridInterpolator 
import numpy as np
import plotly.graph_objects as go

class EntryCondition(Block):
    def __init__(self, name: str, entrySignal) -> None:
        super().__init__(name, [entrySignal], [])

        self.label = 'EntryCond'

        self.node = [
            {'data':{'id':self.name,'label':self.label}},
            {'data': {'id': self.name + '_' + name, 'parent': self.name, 'label':'In'}},
        ]

        self.entrySignal = entrySignal
    
    def computeOutput(self):
        pass


class SampleTime(Block):
    def __init__(self, name: str, value) -> None:
        super().__init__(name, [], [name])

        self.label = 'ts'

        self.node = [
            {'data':{'id':self.name,'label':self.label}},
            {'data': {'id': self.name + '_' + name, 'parent': self.name, 'label':'Out'}},
        ]

        self.ready = {i:True for i in self.ready.keys()}

        self.value = value
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.value
    
    def reset(self):
        self.visited = False

class Constant(Block):
    def __init__(self, name: str, value) -> None:
        super().__init__(name, [], [name])

        self.label = str(value)

        self.node = [
            {'data':{'id':self.name,'label':self.label}},
            {'data': {'id': self.name + '_' + name, 'parent': self.name,'label':'Out'}},
        ]

        self.ready = {i:True for i in self.ready.keys()}

        self.value = value
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.value
    
    def reset(self):
        self.visited = False

class Parameter(Block):
    def __init__(self, name, value=None) -> None:
        super().__init__(name, [], [name])
        self.label = name
        self.node = [
            {'data':{'id':self.name,'label':self.label}},
            {'data': {'id': self.name + '_' + name, 'parent': self.name,'label':'Out'}},
        ]

        self.ready = {i:True for i in self.ready.keys()}

        self.calibrationParameters[name] = value

    def computeOutput(self):
        self.outputs[self.name] = self.calibrationParameters[self.name]
    
    def reset(self):
        self.visited = False

class Delay(Block):
    def __init__(self, name, signalIn, signalOut) -> None:
        super().__init__(name, [signalIn], [signalOut])

        self.label = '1/z'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signalIn, 'parent': self.name,'label':'In'}},
                    {'data': {'id': self.name + '_' + signalOut, 'parent': self.name,'label':'Out'}},]

        self.memory = {i:0 for i in [signalOut]}
        self.ready = {i:True for i in [signalIn]}

        self.signalOut = signalOut
        self.signalIn = signalIn

    def computeOutput(self):
        self.outputs = self.memory
        self.memory[self.signalOut] = self.inputs[self.signalIn]
    
    def reset(self):
        self.visited = False
        
class Add(Block):
    def __init__(self, name, pos1,pos2, output) -> None:
        super().__init__(name,[pos1,pos2],[output])
        
        self.label = 'Add'
        
        self.node = [{'data':{'id':self.name,'label':self.label}},
                        {'data': {'id': self.name + '_' + pos1,'label':'+', 'parent': self.name}},
                        {'data': {'id': self.name + '_' + pos2,'label':'+', 'parent': self.name}},
                        {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}},]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] + self.inputs[self.inputs_list[1]]

class Subtract(Block):
    def __init__(self, name, pos, neg, output) -> None:
        super().__init__(name,[pos,neg],[output])
        
        self.label = 'Subtract'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + pos,'label':'+', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + neg,'label':'-', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] - self.inputs[self.inputs_list[1]]

class Min(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name,[signal1,signal2],[output])
        
        self.label = 'Min'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal1,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + signal2,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output,'label':'Out', 'parent': self.name}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.minimum(self.inputs[self.inputs_list[0]], self.inputs[self.inputs_list[1]])

class Max(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name,[signal1,signal2],[output])
        
        self.label = 'Max'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal1,'label':'In', 'parent': self.name, 'label':'In'}},
                    {'data': {'id': self.name + '_' + signal2,'label':'In', 'parent': self.name, 'label':'In'}},
                    {'data': {'id': self.name + '_' + output,'label':'Out', 'parent': self.name, 'label':'Out'}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.maximum(self.inputs[self.inputs_list[0]], self.inputs[self.inputs_list[1]])
    

class Multiply(Block):
    def __init__(self, name, factor1, factor2, output) -> None:
        super().__init__(name,[factor1,factor2],[output])
        
        self.label = 'Multiply'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + factor1,'label':'x', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + factor2,'label':'x', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]]

class Divide(Block):
    def __init__(self, name, numerator, denominator, output) -> None:
        super().__init__(name, [numerator,denominator], [output])
        self.label = 'Divide'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + numerator,'label':'x', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + denominator,'label':'/', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] / (self.inputs[self.inputs_list[1]]+1e-5)

class And(Block):
    def __init__(self, name, signal1, signal2, output) -> None:
        super().__init__(name, [signal1,signal2], [output])
        self.label = 'And'
        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal1,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + signal2,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]]

class Not(Block):
    def __init__(self, name, signal, output) -> None:
        super().__init__(name, [signal], [output])

        self.label = 'Not'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = 1-self.inputs[self.inputs_list[0]]

class Or(Block):
    def __init__(self, name, signal1,signal2, output) -> None:
        super().__init__(name, [signal1,signal2], [output])

        self.label = 'Or'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal1,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + signal2,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] + self.inputs[self.inputs_list[1]] - (self.inputs[self.inputs_list[0]] * self.inputs[self.inputs_list[1]])

class Switch(Block):
    def __init__(self, name, switched, default, condition, output) -> None:
        super().__init__(name, [switched,default,condition], [output])

        self.label = 'Switch'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + switched,'label':'Switched', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + default,'label':'Default', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + condition,'label':'Condition', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[1]]*(1-self.inputs[self.inputs_list[2]]) + self.inputs[self.inputs_list[0]]*self.inputs[self.inputs_list[2]]

class Abs(Block):
    def __init__(self, name, signal) -> None:
        super().__init__(name, [signal], [signal])

        self.label = '|u|'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + signal,'label':'In', 'parent': self.name}},
                    {'data': {'id': self.name + '_' + signal, 'parent': self.name, 'label':'Out'}}]
    
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = np.abs(self.inputs[self.inputs_list[0]])

class GreaterThanOrEqual(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '>='

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + first,'label':'First', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + second,'label':'Second', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] >= self.inputs[self.inputs_list[1]]

class GreaterThan(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '>'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + first,'label':'First', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + second,'label':'Second', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] > self.inputs[self.inputs_list[1]]

class LessThanOrEqual(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '<='

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + first,'label':'First', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + second,'label':'Second', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] <= self.inputs[self.inputs_list[1]]

class LessThan(Block):
    def __init__(self, name, first,second, output) -> None:
        super().__init__(name, [first,second], [output])

        self.label = '<'

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + first,'label':'First', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + second,'label':'Second', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'Out'}}]
        
    def computeOutput(self):
        self.outputs[self.outputs_list[0]] = self.inputs[self.inputs_list[0]] < self.inputs[self.inputs_list[1]]

class Input(Block):
    def __init__(self, name, outputs) -> None:
        super().__init__(name,outputs,outputs)
        
        self.node = [{'data':{'id':self.name,'label':name}},] + [{'data': {'id': self.name + '_' + out, 'parent': self.name, 'label':'Out'}} for out in outputs]
    
    def computeOutput(self):
        self.outputs = self.inputs

class Output(Block):
    def __init__(self, name, inputs) -> None:
        super().__init__(name,inputs,inputs)
        
        self.node = [{'data':{'id':self.name,'label':name}}] + [{'data': {'id': self.name + '_' + out, 'parent': self.name, 'label':'Out'}} for out in inputs]
                    
    def computeOutput(self):
        self.outputs = self.inputs

class Map2D(Block):
    def __init__(self,name,x,y,output,DCMPath):
        super().__init__(name, [x,y], [output])
        
        self.label = '2D Map ' + name.split('_')[-1]

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + x,'label':'x', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + y,'label':'y', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'z'}}]
    
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

        self.node = [{'data':{'id':self.name,'label':self.label}},
                    {'data': {'id': self.name + '_' + x,'label':'x', 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}},
                    {'data': {'id': self.name + '_' + output, 'parent': self.name, 'label':'y'}}]
    
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