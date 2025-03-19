import dash
import dash_cytoscape as cyto
from dash import html
from BaseClasses import Block, Connection
from BaseBlocks import Input, Output, EntryCondition
from termcolor import colored

def entry(func):
    """
    This wrapper checks if the user has added an entry condition. 
    If we don't have entry then the previous output will be returned.
    I have to double check how delayed values are updated when we don't have entry.
    In this implementation they will be updated regardless of entry
    """
    def wrapper(self):
        if self.entryCondition:
            prevOutput = self.outputs
            #here comes one of the worst lines of code i have ever written
            entrySignal = list(list(self.entryCondition.values())[0].inputs.values())[0] 

            func(self)

            self.outputs = {key:value*entrySignal+prevOutput[key]*(1-entrySignal) for key, value in self.outputs.items()}
        else:
            func(self)
        
        return None
    return wrapper

class SimulinkBlock(Block):
    def __init__(self, name, inputs, outputs) -> None:
        super().__init__(name,inputs,outputs)

        self.compiled = False

        self.addBlock(Input('Input',inputs))
        
        self.addBlock(Output('Output',outputs))

    def addBlock(self,block:Block):
        if block.name in self.blocks:
            raise Exception(f'{block.name} is already in the monitor. Please rename or remove the block.')
            
        self.blocks[block.name] = block
        self._blocks.extend(block.node)
        self.calibrationParameters | block.calibrationParameters

    def addConnection(self,name:str,data:str,start:Block,end:Block):
        while name in self.connections:
            print(f'{name} is already in the list of connection names! Adding _ to the end of name for uniqueness.')
            name += '_'
        
        connection = Connection(name,data,start,end)
        
        self.connections[name] = connection
        self._connections.append({'data': {'source': connection.start.name + '_' + data, 'target': connection.end.name + '_' + data, 'label':data}})

        start.addOutput(connection)
        end.addInput(connection)
    
    def addCalibrationParameter(self,parameter:Block):
        self.addBlock(parameter)
        self.calibrationParameters[parameter.name] = parameter
    
    def addEntryCondition(self,entryBlock:EntryCondition):
        self.entryCondition[entryBlock.name] = entryBlock
        self.addConnection(name='entry',data=entryBlock.entrySignal,start=self.blocks['Input'],end=entryBlock)

    def checkConnections(self,connections:list[Connection]):
        def check(connection):
            bueno = True
            if connection.data not in connection.start.outputs.keys():
                print(f'The block {connection.start.name} does not output {connection.data}! Please double check the connection {connection.name}')
                bueno = False

            
            if connection.data not in connection.end.inputs.keys():
                print(f'The block {connection.end.name} does not take {connection.data} as input! Please double check the connection {connection.name}')
                bueno = False
            
            return bueno
        
        return [check(connection) for connection in connections]

    def checkInputsAndOutpus(self,block:Block):
        inputs = {i:False for i in block.inputs.keys()}
        outputs = {i:False for i in block.outputs.keys()}
        bueno = True
        
        if block.name != 'Input':
            if len(block.connections_in) != len(inputs):
                print(f'{block.name} takes {len(inputs)} inputs! Please double check the incoming connections.')
            
            for connection in block.connections_in:
                inputs[connection.data] = True
            
            for name, connected in inputs.items():
                if not connected:
                    print(f'{block.name} is missing input connection for {name}!')
                    bueno = False
                    
        
        if block.name != 'Output':        
            if len(block.connections_out) != len(outputs):
                print(f'{block.name} has {len(outputs)} outputs! Please double check the incoming connections.')        
            
            for connection in block.connections_out:
                outputs[connection.data] = True        

            for name, connected in outputs.items():
                if not connected:
                    print(f'{block.name} is missing output connection for {name}!')
                    bueno = False
        
        return bueno

    def compileBlock(self):
        todos_bien = []
        for name,block in self.blocks.items():
            if name == 'Input':
                todos_bien.extend(self.checkConnections(block.connections_out))
                continue

            elif name == 'Output':
                todos_bien.extend(self.checkConnections(block.connections_in))
                continue
            
            todos_bien.extend(self.checkConnections(block.connections))
            
        block_bueno = [self.checkInputsAndOutpus(block) for block in self.blocks.values()]

        todos_bien.extend(block_bueno)

        if all(todos_bien):
            print(colored(f'{self.name} compiled successfully!','green'))
            self.compiled = True

        else:
            self.compiled = False
            print(colored(f'Failed to compile {self.name}!','red'))
    
    @entry
    def computeOutput(self):
        try:
            if not self.compiled:
                raise Exception()
        except:
            print(colored('You must compile the block first!','red'))
        
        outputBlock = self.blocks['Output']
        self.backProp(outputBlock)

        for block in self.blocks.values():
            block.reset()
        
        self.outputs = outputBlock.outputs
    
    def backProp(self,block:Block):
        #this can probably be more efficient
        if block.visited:
            return
        
        if block.name == 'Input':
            block.inputs = self.inputs
            block.computeOutput()
            block.transferData()
            block.visited = True
            return 
        
        if sum(block.ready.values()) == len(block.ready):
            block.computeOutput()
            block.transferData()
            block.visited = True
            return 
        
        for connection in block.connections_in:
            if block.ready[connection.data]:
                continue
            else:
                self.backProp(connection.start)

        block.computeOutput()
        block.transferData()
        block.visited = True
        
        return 
    
    def setInput(self,inputs):
        self.inputs = inputs
    
    def setCalibration(self,**kwargs):
        for block in self.blocks.values():
            for key,value in kwargs.items():
                if key in block.calibrationParameters:
                    block.calibrationParameters[key] = value
                    print(f'{key} calibrated to {value}')
        
    def visualizeModel(self):
        try:
            if not self.compiled:
                raise Exception()
        except:
            print(colored('You must compile the block first!','red'))
        
        app = dash.Dash(__name__)

        app.layout = html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'preset'}, 
            style={'width': '100%', 'height': '1000px'},
            elements=self._blocks + self._connections,
            stylesheet=[
                {
                    'selector': 'preset',
                    'style': {
                        'width': 50, 'height': 50, 'shape': 'rectangle',
                        'background-color': 'white', 'label': 'data(label)',
                        'text-valign': 'bottom', 'text-halign': 'center',
                        'color': 'white', 'border-width': 2, 'border-color': 'black'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
                        'arrow-scale': 1.5, 'width': 2,'label': 'data(label)','color': 'white'
                    }
                },
                {
                    'selector': '[parent]', 
                    'style': {'width': 20, 'height': 20, 'background-color': 'black','text-valign': 'center', 'text-halign': 'center'}
                }
            ]
        )
    ])
        app.run_server(debug=True)