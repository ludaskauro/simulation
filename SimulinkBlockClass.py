import dash
import dash_cytoscape as cyto
from dash import html
from BaseClasses import Block,Connection
from BaseBlocks import Input, Output, EntryCondition
from termcolor import colored
import numpy as np
import pickle
import os
from scipy.optimize import minimize, Bounds
from decorators import entry


class SimulinkBlock(Block):
    def __init__(self, name, raster, inputs, outputs) -> None:
        super().__init__(name,inputs,outputs)

        self.raster = raster

        self.compiled = False

        self.addBlock(Input('Input',inputs))
        
        self.addBlock(Output('Output',outputs))

        self.app = dash.Dash(__name__)

    def addBlock(self,block:Block):
        if block.name in self.blocks:
            raise Exception(f'{block.name} is already in the monitor. Please rename or remove the block.')
            
        self.blocks[block.name] = block
        self.calibrationParameters | block.calibrationParameters

        block.raster = self.raster

    def addConnection(self,name:str,data:str,start:Block,end:Block):
        
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

        self.addConnection(name=f'Input->{entryBlock.entrySignal}->Entry',data=entryBlock.entrySignal,start=self.blocks['Input'],end=entryBlock)
        self._blocks.extend(entryBlock.node)
        self.addBlock(entryBlock)
    
    def saveBlock(self,folder:str='SavedBlocks/')->None:
        """
        Saves a block

        Args:
            folder (str): Path to folder where to save the block. Defaults to 'SavedBlocks/'.
        """
        os.makedirs(folder, exist_ok=True)

        with open(folder+self.name+'.pkl','wb') as f:
            pickle.dump(self,f)
    
    @staticmethod
    def loadBlock(name:str,folder:str='SavedBlocks/'):
        """
        Loads a saved block.

        Args:
            name (str): Name of the saved block
            folder (str): Path to the folder where the block is saved. Defaults to 'SavedBlocks/'.

        Returns:
            block (SimulinkBlock): The saved block.
        """
        os.makedirs(folder, exist_ok=True)
        if name+'.pkl' not in os.listdir('SavedBlocks/'):
            raise Exception(f'Simulink block {name} does not exist')
        
        with open(folder+name+'.pkl','rb') as f:
            block = pickle.load(f)

        return block

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
        
        return [check(connection) for connection in connections.values()]

    def checkInputsAndOutpus(self,block:Block):
        inputs = {i:False for i in block.inputs.keys()}
        outputs = {i:False for i in block.outputs.keys()}
        bueno = True
        
        if block.name != 'Input':
            for connection in block.connections_in.values():
                inputs[connection.data] = True
            
            for name, connected in inputs.items():
                if not connected:
                    print(f'{block.name} is missing input connection for {name}!')
                    bueno = False
        
        if block.name != 'Output':        
            for connection in block.connections_out.values():
                outputs[connection.data] = True        

            for name, connected in outputs.items():
                if not connected:
                    print(f'{block.name} is missing output connection for {name}!')
                    bueno = False

        return bueno
    
    def positionNodes(self):
        
        k = 0.01
        r = 15

        blocks = [b for b in self.blocks.values() if b.name not in ['Input', 'Output']]
        all_blocks = list(self.blocks.values())
        x0 = [b.x for b in blocks]
        y0 = [b.y for b in blocks]

        coords0 = x0+y0
        n_blocks = len(blocks)

        lb = [200]*len(x0) + [-300]*len(y0)
        ub = [1000]*len(x0) + [300]*len(y0)

        constrain = Bounds(lb=lb,ub=ub)

        def getHookeForce(block:Block)->float:
            x = block.x
            y = block.y

            in_x = np.array([connection.start.x for connection in block.connections_in.values()])
            in_y = np.array([connection.start.y for connection in block.connections_in.values()])

            out_x = np.array([connection.start.x for connection in block.connections_out.values()])
            out_y = np.array([connection.start.y for connection in block.connections_out.values()])

            force = -k*np.sum(np.sqrt((in_x-x)**2 + (in_y-y)**2)) - k*np.sum(np.sqrt((out_x-x)**2 + (out_y-y)**2))

            return force
        
        def getRepulseForce(block1:Block, block2:Block)->float:
            x1 = block1.x
            x2 = block2.x

            y1 = block1.y
            y2 = block2.y

            force = r*np.sqrt((x1-x2)**2 + (y1-y2)**2)/(len(blocks)-1)

            return force
        
        def objectiveFunction(coords):
            x = coords[:n_blocks]
            y = coords[n_blocks:]
            for xi,yi,bi in zip(x,y,blocks):
                bi.x = xi
                bi.y = yi
            
            hookes = np.array([getHookeForce(block) for block in blocks])
            repulse = np.array([np.sum([getRepulseForce(block,block2) for block2 in all_blocks if block.name != block2.name]) for block in blocks])

            loss = -repulse - hookes

            return np.sum(loss)
        
        res = minimize(objectiveFunction,coords0, method='trust-constr', options={'maxiter': 50},bounds=constrain)
        x = res.x[:n_blocks]
        y = res.x[n_blocks:]
        for block, xi, yi in zip(blocks,x,y):
            block.x = xi
            block.y = yi
        
    def createConnections(self,block:Block):
        if block.name == 'Input':
            return 
        
        if block.visited:
            return
        
        block.visited = True
        
        for i,inputs in enumerate(block.inputs.keys()):
            for b in self.blocks.values():
                if b.name == block.name or b.name == 'Output':
                    continue

                connection_name = f'{b.name}->{inputs}->{block.name}'
                connection_name_inv = f'{block.name}->{inputs}->{b.name}'

                if inputs in b.outputs.keys() and connection_name_inv not in self.connections and connection_name not in self.connections:
                    self.addConnection(name=connection_name,data=inputs,start=b,end=block)
                    
        for connection in block.connections_in.values():
            self.createConnections(connection.start)
        
        return

    def compileBlock(self,printResult=True):
        self.blocks['Output'].x = 1200
        self.blocks['Output'].y = 0

        self.blocks['Input'].x = 100
        self.blocks['Input'].y = 0

        self.createConnections(self.blocks['Output'])
        for block in self.blocks.values():
            block.reset()

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
            self.positionNodes()

            for block in self.blocks.values():
                block.buildNode()
                self._blocks += block.node
                if self.entryCondition and block.name not in ['Input','Output']:
                    block.entryCondition = self.entryCondition
            
            self.compiled = True
            if printResult:
                print(colored(f'{self.name} compiled successfully!','green'))

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
        if block.visited:
            return
        
        block.visited = True
        
        if block.name == 'Input':
            block.inputs = self.inputs
            block.computeOutput()
            block.transferData()
            block.visited = True
            return 
        
        for connection in block.connections_in.values():
            if block.ready[connection.data] and connection.start.visited:
                continue
            else:
                self.backProp(connection.start)

        block.computeOutput()
        block.transferData()
        
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
        

        self.app.layout = html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'preset'}, 
            style={'width': '100%', 'height': '1000px'},
            elements=self._blocks + self._connections,
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'width': 50, 'height': 50, 'shape': 'rectangle',
                        'background-color': '#c2c4c3', 'label': 'data(label)',
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
                    'style': {'width': 10, 'height': 10, 
                              'background-color': 'black','text-valign': 'bottom', 
                              'text-halign': 'center', 'color':'black'}
                }
            ]
        )
    ])
        self.app.run_server(debug=True)
        
    def __getitem__(self,name:str)->Block:
        return self.blocks[name]

    def getSummary(self):
        info = ''

        for block in self.blocks.values():
            info += f'Block name: {block.name} \n'
            info += f'  Type: {type(block).__name__} \n'
            
            info += f'  Connections: \n'
            connections = block.connections_in | block.connections_out
            if connections:
                for connection in connections.keys():
                    info += f'      {connection} \n'
            else:
                info += '       None \n'
                
            if block.calibrationParameters:
                info += f'  Calibration parameters: \n'
                for name, param in block.calibrationParameters.items():
                    if param is not None:
                        info += f'      {name} is calibrated to {param} \n'
                    else:
                        info += f'      {name} is not calibrated \n'
            
            info += '\n----------------------------------------------\n'

        print(info)