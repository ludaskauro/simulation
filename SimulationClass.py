import pandas as pd
import numpy as np
from SimulinkBlockClass import SimulinkBlock
import pyarrow as pa
import pyarrow.parquet as pq
import os
import polars as pl
from functools import lru_cache
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
from decorators import vectorize
from termcolor import colored

class Simulation:
    def __init__(self,block:SimulinkBlock,data:pd.DataFrame,alreadySimulated:bool=False) -> None:
        """
        This class runs simulations and visualizes the result.

        Args:
            block (SimulinkBlock): The simulink block to be simulated
            data (pd.DataFrame): Data to simulate
            alreadySimulated (bool, optional): Whether a simulation has already been ran. Set to True if you only want to visualize a simulation. Defaults to False.
        """
        self.simulinkBlock = block
        self.data = data

        self.alreadySimulated = alreadySimulated
    
    @vectorize
    def simGenerator(self,data):
        """
        Creates a generator which processes and returns one time step of data at the time. The output is a dictionary with keys for each output signal

        Args:
            data (pandas time series or dictionary): A row of the data from self.data.
        """
        self.simulinkBlock.setInput(data)
        self.simulinkBlock.computeOutput()
    
    def runSimulation(self):      
        """
        This method runs a simulation. Chunks of data are processed and written to parquet files.
        """  
        if not self.simulinkBlock.compiled:
            print(colored('You must compile the block first!','red'))
            return
        #Create a generator from which to process data
        gen = self.simGenerator()
        
        #Extract all names and values of the calibration parameters
        names = list(self.simulinkBlock.calibrationParameters.keys())
        values = [param.value for param in self.simulinkBlock.calibrationParameters.values()]
        
        #set up the column indices of the pandas dataframe which will hold the chunk of processed data
        if len(names) > 1:
            column_index = pd.MultiIndex.from_product(values,names=names) #multiple column indices if several calibration parameters
        elif len(names) < 2 and values:
            column_index = pd.Index(values[0],name=names[0]) #In case there is only one calibration parameter
        elif not values:
            column_index = pd.Index([1]) #no calibration parameters
        
        chunk_size = 5_000 #chunk size to process and store

        #One dataframe for each output signal from the simulink block
        out = {name:pd.DataFrame(np.nan, index=np.arange(chunk_size), columns=column_index) for name in self.simulinkBlock.outputs.keys()}

        first_chunk = {name:True for name in self.simulinkBlock.outputs.keys()}
        writer = {name:None for name in self.simulinkBlock.outputs.keys()}
        
        for output in self.simulinkBlock.outputs.keys():
            os.makedirs(f'outputs/{self.simulinkBlock.name}/{output}', exist_ok=True) #create subfolders for data storage
        
        j = 0

        #Start looping through the data
        for i,step in tqdm(enumerate(gen)):            
            for signal in step.keys():
                try: #I think this catches if the data is not a vector
                    data = step[signal].flatten().reshape(1,-1)
                except:
                    data = step[signal]

                out[signal].iloc[j] = data
                
                if (i+1)%chunk_size == 0 and i != 0:# when we have processed a complete chunk we write to file
                    table = pa.Table.from_pandas(out[signal], preserve_index=False)
                    
                    if first_chunk[signal]: #if this is the first chunk being written we create the writers
                        writer[signal] = pq.ParquetWriter(f'outputs/{self.simulinkBlock.name}/{signal}/output.parquet', table.schema, compression='SNAPPY')
                        first_chunk[signal] = False

                    writer[signal].write_table(table)
                    
                    out[signal] = out[signal].map(lambda x: np.nan) #once the chunk is written set dataframe to nan
                                 
            
            if (i+1)%chunk_size == 0 and i != 0:
                j = 0
            else:
                j += 1
                
        for signal,w in writer.items(): #Process final potentially incomplete chunk
            out[signal] = out[signal].dropna()

            table = pa.Table.from_pandas(out[signal], preserve_index=False)
            
            writer[signal].write_table(table)
            out[signal] = None
            
            if w:
                w.close()
        
        self.alreadySimulated = True

    def buildQuery(self,lst:list):
        """Builds a query in the format necessary to read data.

        Args:
            lst (list): list of queried parameters

        Returns:
            str: string in the required format. for instance, lst = [1,2,3] -> query = "'(1,2,3)'"
        """
        query = "("

        for i in lst:
            i = str(i)
            query += "'{}', ".format(i)
            
        query = query[:-2] if len(lst) > 1 else query[:-1]
        query += ")"

        return query

    @lru_cache(10) #chache 10 latest queries to speed up repetitive queries
    def queryResults(self,**kwargs):
        """
        Query the simulation result. Parse a value for each calibration parameter. Returns a dictionary with keys for each output signal and
        signal values for entered calibration.
        Example:
        >>> result = queryResult(parameter1=1,
        >>>             parameter2=2,
        >>>             parameter3=3)
        >>> print(result) -> {'out1':[1,1,1,1,1], 'out2':[1,3,12,4,5]}
        
        Returns:
            dict: keys are the name of output signals and values are signal values
        """
        if kwargs: 
            parameters = self.simulinkBlock.calibrationParameters.keys()
            sorted_kwargs = {param:kwargs[param] for param in parameters} #sorts the parameters to match the parquet file parameter orde

            query = self.buildQuery(list(sorted_kwargs.values())) #Get the query string
            
            folder_path = f'outputs/{self.simulinkBlock.name}/' 
            
            #Get the result using polars (using poolars to increase performance)
            outputs = pl.DataFrame({key:pl.scan_parquet(folder_path+key+'/output.parquet').select([query]).collect() for key in self.simulinkBlock.outputs.keys()})
        
        else: #when there are no calibrated parameters
            folder_path = f'outputs/{self.simulinkBlock.name}/'
            outputs = pl.DataFrame({key:pl.scan_parquet(folder_path+key+'/output.parquet').collect() for key in self.simulinkBlock.outputs.keys()})

        return outputs
    
    def showQuery(self,**kwargs):
        """
        Plots a query. Each output signal is plotted in an interactive line subplot. Parse a value for each calibration parameter.
        """
        query = self.queryResults(**kwargs) #Get the query
        fig = make_subplots(rows=query.shape[1], cols=1, subplot_titles=query.columns, shared_xaxes=True)

        for i, col in enumerate(query.columns):
            fig.add_trace(go.Scatter(y=query[col],mode='lines'),row=i+1,col=1)
        
        fig.update_layout(height=300*len(query.columns))
        fig.show()
    
    def showMax(self,signal):
        """
        Show polar coordinate plot of the max value of entered signal. This is a way to visualize all calibration parameters influence on max value.

        Args:
            signal (str): Name of signal to analyse
        """
        folder_path = f'outputs/{self.simulinkBlock.name}/'

        parameters = {key:self.simulinkBlock.calibrationParameters[key].outputs[key].flatten() for key in self.simulinkBlock.calibrationParameters.keys()}
        
        parameter_names = list(parameters.keys())
        
        df = pl.scan_parquet(folder_path+signal+'/output.parquet')

        max_array = df.max().collect().transpose() #get the max value of requested signal
        max_array.columns = [signal]

        columns = df.collect_schema().names()
        colVal = []
        for col in columns:
            colVal.append(re.findall(r'[-+]?(?:\d*\.*\d+)', col)) 

        tmp = pl.DataFrame(colVal).transpose().cast(pl.Float64) #create new dataframe with correct structure for plot
        
        tmp.columns = parameter_names
        tmp = tmp.with_columns(max_array)
        
        #Plot
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = tmp[signal],
                        colorscale = ['#63520D',
                                    '#7A6200',
                                    '#856A00',
                                    '#8F7200',
                                    '#997A00',
                                    '#A38300',
                                    '#AD8B00',
                                    '#B89300',
                                    '#C29B00',
                                    '#CCA300',
                                    '#D6AB00',
                                    '#E0B400',
                                    '#EBBC00',
                                    '#F5C400',
                                    '#FFCC00',
                                    '#FFCE0A',
                                    '#FFD014',
                                    '#FFD21F',
                                    '#FFD429',
                                    '#FFD633',
                                    '#FFD83D',
                                    '#FFDA47',
                                    '#FFDC52',
                                    '#FFDE5C'],
                        showscale = True,
                        ),
                dimensions=[
                    dict(label=param, values=tmp[param])
                    for param in parameter_names if tmp[param].n_unique() > 1
                ]+[dict(label=signal, values=tmp[signal])]
            )
        )

        fig.update_layout(font=dict(size=20))
        fig.update_layout(height=800, title=f'Max values for {signal}')
        fig.show()
    
    def showMin(self,signal):
        """
        Show polar coordinate plot of the min value of entered signal. This is a way to visualize all calibration parameters influence on min value.

        Args:
            signal (str): Name of signal to analyse
        """
        folder_path = f'outputs/{self.simulinkBlock.name}/'

        parameters = {key:self.simulinkBlock.calibrationParameters[key].outputs[key].flatten() for key in self.simulinkBlock.calibrationParameters.keys()}
        
        parameter_names = list(parameters.keys())
        
        df = pl.scan_parquet(folder_path+signal+'/output.parquet')

        min_array = df.min().collect().transpose() #get the min value of requested signal
        min_array.columns = [signal]

        columns = df.collect_schema().names()
        colVal = []
        for col in columns:
            colVal.append(re.findall(r'[-+]?(?:\d*\.*\d+)', col))

        tmp = pl.DataFrame(colVal).transpose().cast(pl.Float64) #create new dataframe with correct structure for plot
        tmp.columns = parameter_names
        tmp = tmp.with_columns(min_array)

        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = tmp[signal],
                        colorscale = [ '#63520D',
                                    '#7A6200',
                                    '#856A00',
                                    '#8F7200',
                                    '#997A00',
                                    '#A38300',
                                    '#AD8B00',
                                    '#B89300',
                                    '#C29B00',
                                    '#CCA300',
                                    '#D6AB00',
                                    '#E0B400',
                                    '#EBBC00',
                                    '#F5C400',
                                    '#FFCC00',
                                    '#FFCE0A',
                                    '#FFD014',
                                    '#FFD21F',
                                    '#FFD429',
                                    '#FFD633',
                                    '#FFD83D',
                                    '#FFDA47',
                                    '#FFDC52',
                                    '#FFDE5C'],
                        showscale = True,
                        ),
                dimensions=[
                    dict(label=param, values=tmp[param])
                    for param in parameter_names if tmp[param].n_unique() > 1
                ]+[dict(label=signal, values=tmp[signal])]
            )
        )

        fig.update_layout(height=800, title=f'Min values for {signal}')
        fig.show()
    
    def setCalibration(self,**kwargs):
        """
        Set a calibration.
        """
        self.simulinkBlock.setCalibration(**kwargs)
    
    def getCalibrationParameters(self):
        """
        Show calibration name
        """
        print('Calibratable parameters:',*list(self.simulinkBlock.calibrationParameters.keys()))
    
    def getCalibrationInfo(self):
        """
        Get calibration info, name and values of all calibration parameters.
        """
        self.simulinkBlock.getCalibrationInfo()

