import pandas as pd
import numpy as np
from SimulinkBlockClass import SimulinkBlock
import pyarrow as pa
import pyarrow.parquet as pq
import os
import polars as pl
from functools import lru_cache

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

            outputShape[i] = param.shape[0]
        
        self.outputShape = outputShape

        def reshape(value):
            return np.full(outputShape,value)
        
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
        
        out = {name:None for name in self.simulinkBlock.outputs.keys()}

        first_chunk = {name:True for name in self.simulinkBlock.outputs.keys()}
        writer = {name:None for name in self.simulinkBlock.outputs.keys()}
        
        for output in self.simulinkBlock.outputs.keys():
            os.makedirs(f'outputs/{self.simulinkBlock.name}/{output}', exist_ok=True)

        for i,step in enumerate(gen):
            for signal in step.keys():
                data = step[signal].flatten().reshape(1,-1)
                df_tmp = pd.DataFrame(data,columns=index)
                out[signal] = pd.concat([out[signal], df_tmp],axis=0)
            
                if i%500 == 0:
                    table = pa.Table.from_pandas(out[signal], preserve_index=False)
                    
                    if first_chunk[signal]:
                        writer[signal] = pq.ParquetWriter(f'outputs/{self.simulinkBlock.name}/{signal}/output.parquet', table.schema, compression='SNAPPY')
                        first_chunk[signal] = False

                    writer[signal].write_table(table)
                    out[signal] = None
        
        for name,w in writer.items():
            table = pa.Table.from_pandas(out[name], preserve_index=False)
            
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
    
    def setCalibration(self,**kwargs):
        self.simulinkBlock.setCalibration(**kwargs)
    
    def getCalibrationParameters(self):
        print('Calibratable parameters:',*list(self.simulinkBlock.calibrationParameters.keys()))
    
    def getCalibrationInfo(self):
        self.simulinkBlock.getCalibrationInfo()