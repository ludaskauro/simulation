import pickle
import os
from abc import ABC, abstractmethod

class Connection:
    def __init__(self,name:str,data:str,start,end) -> None:
        self.name = name
        self.data = data
        self.start = start
        self.end = end
    
    def transferData(self):
        self.end.inputs[self.data] = self.start.outputs[self.data]
        self.end.ready[self.data] = True

class Block(ABC):
    def __init__(self,name:str,inputs:list[str],outputs:list[str]) -> None:
        self.name:str = name

        self.label:str = ''

        self.inputs_list:list[str] = inputs
        self.outputs_list:list[str] = outputs

        self.inputs:dict = {i:0 for i in inputs}
        self.outputs:dict = {i:0 for i in outputs}

        self.connections_in:list[Connection] = []
        self.connections_out:list[Connection] = []

        self.ready:dict = {i:False for i in inputs}
        self.visited:bool = False

        self.calibrationParameters:dict = {}

        self.blocks:dict = {}
        self.connections:dict = {}

        self._connections:list[Connection] = []
        self._blocks:list[Block] = []

        self.node:dict = {}

        self.entryCondition:dict = {}
    
    def addInput(self,connection:Connection):
        if len(self.connections_in) > len(self.inputs):
            raise Exception(f'More than the allowed number of inputs are connected to {self.name}')
        
        self.connections_in.append(connection)
    
    
    def addOutput(self,connection:Connection):        
        self.connections_out.append(connection)
    
    @abstractmethod
    def computeOutput(self):
        pass

    def setCalibration(self):
        pass

    def addCalibrationParameter(self):
        pass

    def getCalibrationInfo(self):
        if self.calibrationParameters:
            print(f'Calibration parameters for {self.name}:')
            for name, param in self.calibrationParameters.items():
                for p,v in param.calibrationParameters.items():
                    if v is None:
                        print(f'    {p} is not calibrated')
                    else:
                      print(f'    {p} is calibrated to {v.squeeze()}')
        else:
            print(f'{self.name} has calibration parameters')

    def saveBlock(self,folder='inhouseBlocks/'):
        os.makedirs(folder, exist_ok=True)

        with open(folder+self.name+'.pkl','wb') as f:
            pickle.dump(self,f)
    
    @staticmethod
    def loadBlock(name,folder='inhouseBlocks/'):
        os.makedirs(folder, exist_ok=True)
        if name+'.pkl' not in os.listdir('inhouseBlocks/'):
            raise Exception(f'Simulink model {name} does not exist')
        
        with open(folder+name+'.pkl','rb') as f:
            model = pickle.load(f)

        return model

    def transferData(self):
        for connection in self.connections_out:
            connection.transferData()

    def reset(self):
        self.ready = {i:False for i in self.ready.keys()}
        self.visited = False