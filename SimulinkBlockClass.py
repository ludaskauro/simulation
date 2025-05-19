import dash
import dash_cytoscape as cyto
from dash import html
from BaseBlocks import Input, Output, EntryCondition, Delay, Parameter
from BaseClasses import Connection, Block
from decorators import entry
from termcolor import colored
import numpy as np
import pickle
import os
from scipy.optimize import minimize, Bounds


class SimulinkBlock(Block):
    def __init__(self, name:str, raster:float, inputs:list[str], outputs:list[str]) -> None:
        """
        This class represents a simulink block. By initialising a simulink block and adding blocks, entry conditions and calibration parameters, 
        we can build complex models through which we can parse data to analyse behaviours and calibrations.

        Inherits from BaseClasses.Block

        Args:
            name (str): The name of the block. This will be used for storing simulation results and can be useful in fault tracing.
            raster (float): What raster the monitor runs in. 10ms -> 0.01
            inputs (list[Str]): The input signals for the monitor.
            outputs (list[Str]): The output signals for the monitor
        
        Methods:
            addBlock (block): add a block
            addConnection (name,data,start,end): used during compilation of block to create connections between blocks
            addCalibrationParameter (parameter): add a calibration parameter
            addEntryCondition (entryBlock): add a entry condition
            saveBlock (folder): pickle and save a simulink block
            loadBlock (name,folder): static method for loading pickled simulink block
            checkConnections (connections): used during compilation of block to check for invalid or missing connections
            checkInputsAndOutpus (block): used during compilation of block to check for invalid or missing inputs/outputs
            positionNodes (): used during compilation of block to position internal blocks in the graph
            createConnections (block): used during compilation of block to create connections between blocks
            compileBlock (printResult): should always be called before passing data through block! creates and checks all connections, positions nodes and determines order of computation
            reset (): resets the block
            findCycles (): used during compilation of block to identify cycles in the graph
            topologicalSort (): used during compilation of block to determine computation order
            depthFirstSearch (): used in topologicalSort to determine computation order
            computeOutput (): computes output
            setInput (inputs): sets the input
            setCalibration (kwargs): sets values to the calibration parameters entered in kwargs
            getCalibrationInfo (): prints all calibration parameters and their values
            visualizeBlock (): plots an interactive graph showing the compiled simulink model
            getSummary (): prints all blocks and what inputs/outputs they recieve/output
        """

        super().__init__(name,inputs,outputs)
        
        self.raster = raster

        self.compiled = False

        self.addBlock(Input('Input',inputs))
        
        self.addBlock(Output('Output',outputs))

        self.app = dash.Dash(__name__)

    def addBlock(self,block:Block):
        """
        Adds a block to the model. The raster and calibration parameters are shared to the new block.

        Args:
            block (Block): Any block which inherits from BaseClasses.Block

        Raises:
            Exception: Multiple blocks with the same name are not allowed
        """
        if block.name in self.blocks:
            raise Exception(f'{block.name} is already in the monitor. Please rename or remove the block.')
            
        self.blocks[block.name] = block
        self.calibrationParameters = self.calibrationParameters | block.calibrationParameters

        block.raster = self.raster

    def addConnection(self,name:str,data:str,start:Block,end:Block):
        """
        Adds a connection between two blocks which can transfer data between them.

        Args:
            name (str): Name of the connection
            data (str): name of signal being transfered across the connection
            start (Block): Start block of the connection
            end (Block): End block of the connection
        """
        
        connection = Connection(name,data,start,end)
        
        self.connections[name] = connection
        self._connections.append({'data': {'source': connection.start.name + '_' + data, 'target': connection.end.name + '_' + data, 'label':data}}) #Adds to the interactive dash graph

        #Sets up input and output of blocks
        start.addOutput(connection) 
        end.addInput(connection)
    
    def addCalibrationParameter(self,parameter:Parameter):
        """
        Adds a calibration parameter to the block. This must be of type Parameter. Calibration parameters values can be changed to perform different calibrated simulations

        Args:
            parameter (Parameter): Block of type Parameter
        """
        
        self.addBlock(parameter)

        if parameter.name not in self.calibrationParameters:
            self.calibrationParameters[parameter.name] = parameter
    
    def addEntryCondition(self,entryBlock:EntryCondition):
        """
        Adds a global entry condition which is applied to all blocks in the model

        Args:
            entryBlock (EntryCondition): Block of type EntryCondition
        """
        self.entryCondition[entryBlock.name] = entryBlock

        self.addConnection(name=f'Input->{entryBlock.entrySignal}->Entry',data=entryBlock.entrySignal,start=self.blocks['Input'],end=entryBlock) #add connection from input to the block
        self._blocks.extend(entryBlock.node) #adds the block to the interactive graph
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

    def checkConnections(self,connections:list[Connection]) -> list[bool]:     
        """
        Checks whether all connection are connected to valid blocks with correct input/output

        Args:
            connections (list[Connection]): list of connections in the block

        Returns:
            list[bool]: which connections are valid
        """

        def check(connection):
            #checks whether the signal transfered is in the inputs/outputs of connected blocks
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
        """Checks validity of inputs and outputs to a block.

        Args:
            block (Block): block to check

        Returns:
            bool: whether the block has valid inputs/outputs
        """
        inputs = {i:False for i in block.inputs.keys()}
        outputs = {i:False for i in block.outputs.keys()}
        bueno = True
        
        if block.name != 'Input':
            #check the inputs for all blocks
            for connection in block.connections_in.values():
                inputs[connection.data] = True
            
            for name, connected in inputs.items():
                if not connected:
                    print(f'{block.name} is missing input connection for {name}!')
                    bueno = False
        
        if block.name != 'Output':        
            #check the outputs for all blocks
            for connection in block.connections_out.values():
                outputs[connection.data] = True        

            for name, connected in outputs.items():
                if not connected:
                    print(f'{block.name} is missing output connection for {name}!')
                    bueno = False

        return bueno
    
    def positionNodes(self):
        """
        This method positions the nodes in the interactive graph by treating connections as springs and balancing Hooke forces with repulsive forces.
        This might not be the most efficient way to do it but i thought it would be fun to try so here we are and it works semi-well
        """
        #calibrated parameters for force scaling
        k = 0.01
        r = 15

        blocks = [b for b in self.blocks.values() if b.name not in ['Input', 'Output']]
        all_blocks = list(self.blocks.values())
        x0 = [b.x for b in blocks] #start coordinates
        y0 = [b.y for b in blocks]

        coords0 = x0+y0
        n_blocks = len(blocks)

        #upper and lower bounds
        lb = [200]*len(x0) + [-300]*len(y0) 
        ub = [1000]*len(x0) + [300]*len(y0)

        constrain = Bounds(lb=lb,ub=ub)

        def getHookeForce(block:Block)->float:
            """
            Generates forces based on Hookes law on one block

            Args:
                block (Block): Any block in the model

            Returns:
                float: the force acting on the block
            """
            x = block.x
            y = block.y

            in_x = np.array([connection.start.x for connection in block.connections_in.values()])
            in_y = np.array([connection.start.y for connection in block.connections_in.values()])

            out_x = np.array([connection.start.x for connection in block.connections_out.values()])
            out_y = np.array([connection.start.y for connection in block.connections_out.values()])

            force = -k*np.sum(np.sqrt((in_x-x)**2 + (in_y-y)**2)) - k*np.sum(np.sqrt((out_x-x)**2 + (out_y-y)**2))

            return force
        
        def getRepulseForce(block1:Block, block2:Block)->float:
            """
            Determines the repulsive force between two blocks, this is just proportional to the distance beetween them

            Args:
                block1 (Block): Any block in the model
                block2 (Block): Any block in the model

            Returns:
                float: the repulsive force
            """
            x1 = block1.x
            x2 = block2.x

            y1 = block1.y
            y2 = block2.y

            force = r*np.sqrt((x1-x2)**2 + (y1-y2)**2)/(len(blocks)-1)

            return force
        
        def objectiveFunction(coords) -> float:
            """
            Objective function to minimize

            Args:
                coords (_type_): coordinates

            Returns:
                float: objective function value
            """
            x = coords[:n_blocks]
            y = coords[n_blocks:]
            for xi,yi,bi in zip(x,y,blocks):
                bi.x = xi
                bi.y = yi
            
            hookes = np.array([getHookeForce(block) for block in blocks])
            repulse = np.array([np.sum([getRepulseForce(block,block2) for block2 in all_blocks if block.name != block2.name]) for block in blocks])

            loss = -repulse - hookes

            return np.sum(loss)
        
        #minimize the objective function
        res = minimize(objectiveFunction,coords0, method='trust-constr', options={'maxiter': 50},bounds=constrain)
        x = res.x[:n_blocks]
        y = res.x[n_blocks:]

        #Set optimized coordinates
        for block, xi, yi in zip(blocks,x,y):
            block.x = xi
            block.y = yi
        
    def createConnections(self,block:Block):
        """
        Recursively create all connections between all blocks in the model

        Args:
            block (Block): Any block in the model
        """
        #We don't want to process the Input block or previously processed blocks
        if block.name == 'Input':
            return 
        
        if block.visited:
            return
        
        block.visited = True
        
        for i,inputs in enumerate(block.inputs.keys()): #loop through the inputs to the blocks and find all blocks which outputs the signal. Create connection to these blocks
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

    def compileBlock(self,printResult:bool=True):
        """
        This method compiles the block. It is necessary to run when a new block has been created.
        1. All connections are created and checked.
        2. All inputs and outputs to all blocks are checked.
        3. The blocks are positioned in the graph.
        4. Shares entry conditions between all blocks.
        5. Performs topological sort to determine in which order all blocks will compute their output and transfer data along connections.

        Args:
            printResult (bool): Whether to print success message or not. Defaults to True.
        """
        #Position output and input blocks
        self.blocks['Output'].x = 1200
        self.blocks['Output'].y = 0

        self.blocks['Input'].x = 100
        self.blocks['Input'].y = 0

        for block in self.blocks.values():
            self.createConnections(block) #create connections for all blocks

        for block in self.blocks.values():
            block.reset() #resets visited booleans

        todos_bien = [] #check all connections
        for name,block in self.blocks.items():
            if name == 'Input':
                todos_bien.extend(self.checkConnections(block.connections_out))
                continue

            elif name == 'Output':
                todos_bien.extend(self.checkConnections(block.connections_in))
                continue
            
            todos_bien.extend(self.checkConnections(block.connections))
        
        #check block inputs/outputs
        block_bueno = [self.checkInputsAndOutpus(block) for block in self.blocks.values()]

        todos_bien.extend(block_bueno)

        if all(todos_bien): #if all is good, position nodes, share entry conditions and determine computation order
            self.positionNodes()

            for block in self.blocks.values():
                block.buildNode()
                self._blocks += block.node
                if self.entryCondition and block.name not in ['Input','Output']:
                    block.entryCondition = self.entryCondition
            
            self.compiled = True

            self.topologicalSort()

            if printResult:
                print(colored(f'{self.name} compiled successfully!','green'))

        else:
            self.compiled = False
            print(colored(f'Failed to compile {self.name}!','red'))
    
    def reset(self):
        """
        Resets all blocks. This sets block.visited to False and wipes memory 
        """
        for block in self.blocks.values():
            block.reset()
        
        self.visited = False
    
    def findCycles(self):
        """
        Identifies cycles in the graph. these needs to be broken up to perform topological sorting

        """
        def dfsCycleDetection(current, stack, visited, stack_set, cycles):
            if visited[current.name]:  # Already visited
                if current.name in stack_set:  # Found a cycle
                    # Extract the cycle by slicing the stack
                    cycle_start_index = stack.index(current.name)
                    cycle = stack[cycle_start_index:]
                    cycles.append(cycle)
                return

            visited[current.name] = True
            stack.append(current.name)
            stack_set.add(current.name)

            for connection in current.connections_out.values():
                dfsCycleDetection(connection.end, stack, visited, stack_set, cycles)

            stack.pop()
            stack_set.remove(current.name)

        visited = {block.name: False for block in self.blocks.values()}
        cycles = []

        for block in self.blocks.values():
            if not visited[block.name]:
                dfsCycleDetection(block, [], visited, set(), cycles)

        # Mark blocks involved in cycles
        for cycle in cycles:
            for block_name in cycle:
                self.blocks[block_name].in_cycle = True
    
    def topologicalSort(self):
        """
        Perform topological sorting to determine order in which to compute blocks outputs.
        """
        order = ['Input']+[name for name in self.entryCondition.keys()]
        call_stack = []

        # Initialize visited status and cycle information
        for block in self.blocks.values():
            if block.name == 'Input' or block.name in self.entryCondition.keys():
                block.visited = True
            else:
                block.visited = False
                block.in_cycle = False  # Assume blocks are not in cycles initially

        # Step 1: Detect cycles and mark blocks involved in cycles
        self.findCycles()

        # Step 2: Regular Depth-First Search with improved skipping logic
        for block in self.blocks.values():
            if block.visited:
                continue
            call_stack = self.depthFirstSearch(block, call_stack)

        order.extend(call_stack[::-1])  # Reverse the stack for topological order
        
        # Save the computation order
        self.computation_order = order

        # Reset visited status for future use
        for block in self.blocks.values():
            block.visited = False

    def depthFirstSearch(self, block:Block, call_stack:list[str]) -> list[str]:
        """
        Perform DFS and breaks cycles

        Args:
            block (Block): Any block in the model
            call_stack (list[str]): order of sorting

        Returns:
            list[str]: order of sorting
        """
        if block.visited:
            return call_stack

        block.visited = True

        for connection in block.connections_out.values():
            # Skip Delay blocks that are part of cycles
            if isinstance(connection.end, Delay) and connection.end.in_cycle and not isinstance(block, Delay):
                continue
            call_stack = self.depthFirstSearch(connection.end, call_stack)

        call_stack.append(block.name)

        return call_stack

    @entry
    def computeOutput(self):
        """
        Computes output by looping through the blocks in the computational order and computing their output and transfering data across connections
        """
        self.blocks['Input'].inputs = self.inputs
        for name in self.computation_order:
            self.blocks[name].computeOutput()
            self.blocks[name].transferData()

        self.outputs = self.blocks['Output'].outputs

    def setInput(self,inputs):
        """
        Sets the input

        Args:
            inputs (_type_): anything with the same kind of functionality for data retrieval as a dictionary or pandas time data frame 
        """
        self.inputs = inputs
    
    def setCalibration(self,**kwargs):
        """
        Sets calibration
        """
        for key,value in kwargs.items():
            if key in self.calibrationParameters:
                self.calibrationParameters[key].value = value 
                print(f'{key} calibrated to {value}')
    
    def getCalibrationInfo(self)->None:
        """
        Prints information of the set calibration
        """
        if self.calibrationParameters:
            print(f'Calibration parameters for {self.name}:')
            for name, param in self.calibrationParameters.items():
                if param.value is None:
                    print(f'    {name} is not calibrated')
                else:
                    print(f'    {name} is calibrated to {np.array(param.value).squeeze()}')
        else:
            print(f'{self.name} has no calibration parameters')

    def visualizeBlock(self):
        """
        Show the interactive graph. This runs through a dash app so the result will be shown on a local host server. in jupyter notebook the graph will be shown as any other plot.

        Raises:
            Exception: The model needs to have been successfully compiled
        """
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
        """
        Overwritten dunder method. simulinkBlock[name] will return the block named "name" from the simulink block

        Args:
            name (str): the name of a block in the model

        Returns:
            Block: block named "name"
        """
        return self.blocks[name]

    def getSummary(self):
        """
        Prints a summary of all blocks, how they are connected and what data is being computed
        """
        info = ''

        for name in self.computation_order:
            block = self.blocks[name]
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