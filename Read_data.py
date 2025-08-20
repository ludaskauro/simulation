import mdfreader
import os
import pandas as pd
from tqdm import tqdm

def ReadINCAFile(directory: str, channel_list: list[str],raster:float):
    """
    Reads selected signals from INCA-measurement files in the format MDF (.dat) into a pandas DataFrame.
    Signals are aligned on the same time axis through merging.
    
    Input:
        directory : str
            Path to the directory with the INCA files

        channel_list : list[str]
            List containing the selected signals (channels) to extract.
    
    Output:
        pd.DataFrame
            Data concatenated into a pandas DataFrame.
    """
    channel_list = list(set(channel_list))
    df = None
    for file_name in tqdm(sorted(os.listdir(directory))):
        
        # Read MDF data file
        try:
            data = mdfreader.Mdf(os.path.join(directory, file_name),channel_list=channel_list)
            data.resample(raster)
            data.convert_to_pandas()
            
            # Get group
            group = [s for s in data.keys() if s.endswith("group")][0]
            data = data[group].dropna(ignore_index=True)
            data['file'] = file_name

            # Add to dataframe
            df = pd.concat([df, data], axis=0)
            
        except Exception as e:
            print(f'Failed to load data from {file_name}')
            print(e)
            continue
    
    df = df.reset_index(drop=True)
    return df