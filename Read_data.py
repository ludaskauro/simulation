import mdfreader
import os
import pandas as pd
from tqdm import tqdm

def ReadINCAFile(directory:str,channel_list:list[str]):
    '''
    Reads selected signals from INCA-measurement files in the format MDF (.dat) into a pandas dataframe. 
    The data from all files will be concatenated into one single dataframe.
    Input:
        directory : str
            Path to directory with the INCA files

        channel_list : list[str]
            list containing the selected signals
    
    Output:
        df : pandas.DataFrame
            data concatenated into one pandas dataframe
    '''

    df = None
    for file_name in tqdm(sorted(os.listdir(directory))):
        
        # Read MDF data file
        try:
            data = mdfreader.Mdf(os.path.join(directory, file_name),channel_list=channel_list)
            data.resample()
            data.convert_to_pandas()
            
            # Get group
            group = [s for s in data.keys() if s.endswith("group")][0]
            data = data[group].dropna(ignore_index=True)
            data['file'] = file_name

            # Add to dataframe
            df = pd.concat([df, data], axis=0)
        except:
            continue
    
    df = df.reset_index(drop=True)
    return df[channel_list]