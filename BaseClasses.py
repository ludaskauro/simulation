from abc import ABC, abstractmethod
import numpy as np

class Connection:
    def __init__(self,name:str,data:str,start,end) -> None:
        """
        Connections create a connection between two blocks. 

        Args:
            name (str): Name for connection. If the connection is created during compiling a SimulinkBlock it will be start->data->end
            data (str): what signal is being transfered across the connection
            start (Block): Start block on the connection
            end (Block): End block on the connection
        """
        self.name = name
        self.data = data
        self.start = start
        self.end = end
    
    def transferData(self)->None:
        """
        Transfers data along the connection
        """
        self.end.inputs[self.data] = self.start.outputs[self.data]
        self.end.ready[self.data] = True

class Block(ABC):
    def __init__(self,name:str,inputs:list[str],outputs:list[str]) -> None:
        """
        The Block class contains all base attributes and methods needed for implementing a simulink block. 

        Args:
            name (str): The name of the block. Useful when fault tracing.
            inputs (list[str]): List containing the input signals.
            outputs (list[str]): List containing the output signals.
        """
        self.name:str = name

        self.label:str = ''

        self.inputs_list:list[str] = inputs
        self.outputs_list:list[str] = outputs

        self.inputs:dict = {i:0 for i in inputs}
        self.outputs:dict = {i:0 for i in outputs}

        self.inputPorts:dict = {i:'In' for i in inputs}
        self.outputPorts:dict = {i:'Out' for i in outputs}

        self.label:str = ''

        self.connections_in:dict = {}
        self.connections_out:dict = {}

        self.ready:dict = {i:False for i in inputs}
        self.visited:bool = False

        self.calibrationParameters:dict = {}

        self.blocks:dict = {}
        self.connections:dict = {}

        self._connections:list[Connection] = []
        self._blocks:list[Block] = []

        self.node:dict = {}

        self.entryCondition:dict = {}

        self.memory:dict = {}

        self.x:int | float | None = 600
        self.y:int | float | None = 0

        self.raster:int | float | None = None
    
    def buildNode(self)->None:
        #raise NotImplementedError('This method is not yet implemented. The idea is to automate the creation of nodes for the visualized graph.')
        n_inputs, n_outputs = len(self.inputPorts), len(self.outputPorts)
        self.node = [{'data':{'id':self.name,'label':self.label}, 'position':{'x':self.x, 'y':self.y}}]\
              + [{'data': {'id': self.name + '_' + i,'label':l, 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}, 'position':{'x':self.x-30, 'y':self.y+n_inputs*20/2-40*j}}  for j,(i,l) in enumerate(self.inputPorts.items())]\
              + [{'data': {'id': self.name + '_' + i,'label':l, 'parent': self.name},'style':{'text-valign': 'bottom', 'text-halign': 'center'}, 'position':{'x':self.x+30, 'y':self.y+n_outputs*20/2-40*j}}  for j,(i,l) in enumerate(self.outputPorts.items())]
    
    def addInput(self,connection:Connection)->None:
        """
        Adds incoming connection to connections_in

        Args:
            connection (Connection): connection to be added
        """

        if len(self.connections_in) > len(self.inputs):
            raise Exception(f'More than the allowed number of inputs are connected to {self.name}')
        
        self.connections_in[connection.name] = connection
    
    def addOutput(self,connection:Connection)->None:
        """
        Adds outgoing connection to connections_out

        Args:
            connection (Connection): connection to be added
        """
        self.connections_out[connection.name] = connection
    
    @abstractmethod
    def computeOutput(self)->None:
        pass
    
    def setCalibration(self)->None:
        raise NotImplementedError('You need to overwrite this method in your class if you want to use them')
    
    def addCalibrationParameter(self)->None:
        raise NotImplementedError('You need to overwrite this method in your class if you want to use them')

    def transferData(self)->None:
        """
        Transfers data from outputs to all downstream connected blocks.
        """
        for connection in self.connections_out.values():
            connection.transferData()

    def reset(self)->None:
        """
        Resets the ready and visited attributes to False. Used after computing output in a SimulinkBlock instance.
        Some BaseBlocks' methods are overwritten.
        """
        self.ready = {i:False for i in self.ready.keys()}
        self.visited = False
    
    def initalize(self)->None:
        """
        Sets all inputs and outputs to 0. Used in beginning of simulations to reset to default settings.
        """
        self.inputs = {signal:0 for signal in self.inputs_list}
        self.outputs = {signal:0 for signal in self.outputs_list}

        if self.blocks.values():
            for block in self.blocks:
                block.initalize()