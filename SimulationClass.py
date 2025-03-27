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

def to_single_feature(param, shape=(-1,1)):
    param = np.array(param)
    if len(param.shape) < 2:
        param = param.reshape(shape)
    return param

def vectorize(func):
    def wrapper(self):
        parameters = self.simulinkBlock.calibrationParameters.items()
        outputShape = [1]*len(parameters)
        for i,(name,param) in enumerate(parameters):
            param = to_single_feature(param.calibrationParameters[name])

            newShape = [1]*len(parameters)
            newShape[i] = -1

            param = param.reshape(newShape)

            self.simulinkBlock.blocks[name].calibrationParameters[name] = param

            outputShape[i] = param.shape[i]
        
        self.outputShape = outputShape

        def reshape(value):
            return np.full(self.outputShape,value)
        
        for i,row in self.data.iterrows():
            row = row.map(reshape)
            func(self,row)

            yield self.simulinkBlock.outputs
            
    return wrapper

class Simulation:
    def __init__(self,sim:SimulinkBlock,data:pd.DataFrame,alreadySimulated:bool=False) -> None:
        self.simulinkBlock = sim
        self.data = data

        self.alreadySimulated = alreadySimulated
    
    @vectorize
    def simGenerator(self,data):
        self.simulinkBlock.setInput(data)
        self.simulinkBlock.computeOutput()
    
    def runSimulation(self):        
        gen = self.simGenerator()
        names = list(self.simulinkBlock.calibrationParameters.keys())
        values = [param.calibrationParameters[name] for name, param in self.simulinkBlock.calibrationParameters.items()]

        index = pd.MultiIndex.from_product(values,names=names)
        
        chunk_size = 5_000

        out = {name:pd.DataFrame(np.nan, index=np.arange(chunk_size), columns=index) for name in self.simulinkBlock.outputs.keys()}

        first_chunk = {name:True for name in self.simulinkBlock.outputs.keys()}
        writer = {name:None for name in self.simulinkBlock.outputs.keys()}
        
        for output in self.simulinkBlock.outputs.keys():
            os.makedirs(f'outputs/{self.simulinkBlock.name}/{output}', exist_ok=True)
        
        j = 0

        for i,step in tqdm(enumerate(gen)):
            
            for signal in step.keys():
                
                if i%chunk_size == 0 and i != 0:
                    table = pa.Table.from_pandas(out[signal], preserve_index=False)
                    
                    if first_chunk[signal]:
                        writer[signal] = pq.ParquetWriter(f'outputs/{self.simulinkBlock.name}/{signal}/output.parquet', table.schema, compression='SNAPPY')
                        first_chunk[signal] = False

                    writer[signal].write_table(table)
                    
                    out[signal] = out[signal].map(lambda x: np.nan)
                    j = 0
                
                data = step[signal].flatten().reshape(1,-1)

                out[signal].iloc[j] = data
                
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

        query = query[:-2]
        query += ")"

        return query

    @lru_cache(10)
    def queryResults(self,**kwargs):
        parameters = self.simulinkBlock.calibrationParameters.keys()
        sorted_kwargs = {param:kwargs[param] for param in parameters}

        query = self.buildQuery(list(sorted_kwargs.values()))
        
        folder_path = f'outputs/{self.simulinkBlock.name}/'

        outputs = pl.DataFrame({key:pl.scan_parquet(folder_path+key+'/output.parquet').select([query]).collect() for key in self.simulinkBlock.outputs.keys()})

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
        max_array.columns = ['max_'+signal]

        columns = df.collect_schema().names()
        colVal = []
        for col in columns:
            colVal.append(re.findall(r'[-+]?(?:\d*\.*\d+)', col))

        tmp = pl.DataFrame(colVal).transpose().cast(pl.Float64)
        
        tmp.columns = parameter_names
        tmp = tmp.with_columns(max_array)

        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = tmp['max_'+signal],
                        colorscale = 'viridis',
                        showscale = True,
                        ),
                dimensions=[
                    dict(label=param, values=tmp[param])
                    for param in parameter_names
                ]+[dict(label='max_'+signal, values=tmp['max_'+signal])]
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
                        colorscale = 'viridis',
                        showscale = True,
                        ),
                dimensions=[
                    dict(label=param, values=tmp[param])
                    for param in parameter_names
                ]+[dict(label='min_'+signal, values=tmp['min_'+signal])]
            )
        )

        fig.update_layout(paper_bgcolor="black")
        fig.update_layout(height=800, title=f'Min values for {signal}')
        fig.show()
    
    def setCalibration(self,**kwargs):
        self.simulinkBlock.setCalibration(**kwargs)
    
    def getCalibrationParameters(self):
        print('Calibratable parameters:',*list(self.simulinkBlock.calibrationParameters.keys()))
    
    def getCalibrationInfo(self):
        self.simulinkBlock.getCalibrationInfo()