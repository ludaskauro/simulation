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
        self.simulinkBlock = block
        self.data = data

        self.alreadySimulated = alreadySimulated
    
    @vectorize
    def simGenerator(self,data):
        self.simulinkBlock.setInput(data)
        self.simulinkBlock.computeOutput()
    
    def runSimulation(self):        
        if not self.simulinkBlock.compiled:
            print(colored('You must compile the block first!','red'))
        
        gen = self.simGenerator()
        
        names = list(self.simulinkBlock.calibrationParameters.keys())
        values = [param.value for name, param in self.simulinkBlock.calibrationParameters.items()]
        
        if len(names) > 1:
            index = pd.MultiIndex.from_product(values,names=names)
        elif len(names) < 2 and values:
            index = pd.Index(values[0],name=names[0])
        elif not values:
            index = pd.Index([1])
        
        chunk_size = 5_000

        out = {name:pd.DataFrame(np.nan, index=np.arange(chunk_size), columns=index) for name in self.simulinkBlock.outputs.keys()}

        first_chunk = {name:True for name in self.simulinkBlock.outputs.keys()}
        writer = {name:None for name in self.simulinkBlock.outputs.keys()}
        
        for output in self.simulinkBlock.outputs.keys():
            os.makedirs(f'outputs/{self.simulinkBlock.name}/{output}', exist_ok=True)
        
        j = 0
        chunk = 0

        for i,step in tqdm(enumerate(gen)):
            
            for signal in step.keys():
                try:
                    data = step[signal].flatten().reshape(1,-1)
                except:
                    data = step[signal]

                out[signal].iloc[j] = data
                
                if i%(chunk_size-1) == 0 and i != 0:
                    table = pa.Table.from_pandas(out[signal], preserve_index=False)
                    
                    if first_chunk[signal]:
                        writer[signal] = pq.ParquetWriter(f'outputs/{self.simulinkBlock.name}/{signal}/output.parquet', table.schema, compression='SNAPPY')
                        first_chunk[signal] = False

                    writer[signal].write_table(table)
                    
                    out[signal] = out[signal].map(lambda x: np.nan)
                                 
            
            if i%(chunk_size-1) == 0 and i != 0:
                j = 0
                chunk += 1
            else:
                j += 1
                
        for signal,w in writer.items():
            out[signal] = out[signal].dropna()

            table = pa.Table.from_pandas(out[signal], preserve_index=False)
            
            writer[signal].write_table(table)
            out[signal] = None
            
            if w:
                w.close()
        
        self.alreadySimulated = True

    def buildQuery(self,lst):
        query = "("

        for i in lst:
            i = str(i)
            query += "'{}', ".format(i)

        query = query[:-2] if len(lst) > 1 else query[:-1]
        query += ")"

        return query

    @lru_cache(10)
    def queryResults(self,**kwargs):
        if kwargs:
            parameters = self.simulinkBlock.calibrationParameters.keys()
            sorted_kwargs = {param:kwargs[param] for param in parameters}

            query = self.buildQuery(list(sorted_kwargs.values()))
            
            folder_path = f'outputs/{self.simulinkBlock.name}/'
            
            outputs = pl.DataFrame({key:pl.scan_parquet(folder_path+key+'/output.parquet').select([query]).collect() for key in self.simulinkBlock.outputs.keys()})
        
        else:
            folder_path = f'outputs/{self.simulinkBlock.name}/'
            outputs = pl.DataFrame({key:pl.scan_parquet(folder_path+key+'/output.parquet').collect() for key in self.simulinkBlock.outputs.keys()})

        return outputs
    
    def showQuery(self,**kwargs):
        query = self.queryResults(**kwargs)
        fig = make_subplots(rows=query.shape[1], cols=1, subplot_titles=query.columns)

        for i, col in enumerate(query.columns):
            fig.add_trace(go.Scatter(y=query[col],mode='lines'),row=i+1,col=1)
        
        fig.show()
    
    def showMax(self,signal):
        folder_path = f'outputs/{self.simulinkBlock.name}/'

        parameters = {key:self.simulinkBlock.calibrationParameters[key].outputs[key].flatten() for key in self.simulinkBlock.calibrationParameters.keys()}
        
        parameter_names = list(parameters.keys())
        
        df = pl.scan_parquet(folder_path+signal+'/output.parquet')

        max_array = df.max().collect().transpose()
        max_array.columns = [signal]

        columns = df.collect_schema().names()
        colVal = []
        for col in columns:
            colVal.append(re.findall(r'[-+]?(?:\d*\.*\d+)', col))

        tmp = pl.DataFrame(colVal).transpose().cast(pl.Float64)
        
        tmp.columns = parameter_names
        tmp = tmp.with_columns(max_array)

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
        folder_path = f'outputs/{self.simulinkBlock.name}/'

        parameters = {key:self.simulinkBlock.calibrationParameters[key].outputs[key].flatten() for key in self.simulinkBlock.calibrationParameters.keys()}
        
        parameter_names = list(parameters.keys())
        
        df = pl.scan_parquet(folder_path+signal+'/output.parquet')

        min_array = df.min().collect().transpose()
        min_array.columns = [signal]

        columns = df.collect_schema().names()
        colVal = []
        for col in columns:
            colVal.append(re.findall(r'[-+]?(?:\d*\.*\d+)', col))

        tmp = pl.DataFrame(colVal).transpose().cast(pl.Float64)
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
        self.simulinkBlock.setCalibration(**kwargs)
    
    def getCalibrationParameters(self):
        print('Calibratable parameters:',*list(self.simulinkBlock.calibrationParameters.keys()))
    
    def getCalibrationInfo(self):
        self.simulinkBlock.getCalibrationInfo()

