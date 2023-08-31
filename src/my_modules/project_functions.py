import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re # for regular expressions
from tqdm import tqdm # for progress bar


def plot_discrete_var(df):
    for col in df.columns:
        plt.figure(figsize=(10,5))
        sns.countplot(data=df, x=col, palette = "crest")

        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.show()
        
def plot_continuous_var(df, bins):
    
    for col in df.columns:
        #defining plot size
        plt.figure(figsize=(10,5))
        
        # Histogram
        # defining subplot 1 
        plt.subplot(1,2,1)
        # sns.histplot(df[col], bins = bins, kde = True, hue = "crest")
        sns.histplot(df[col], bins = bins, kde = True, palette = "crest")
        plt.title(f"Histogram for {col}")
        
        # Boxplot
        plt.subplot(1,2,2)
        sns.boxplot(df[col])
        plt.title(f"Boxplot for {col}")
        
        plt.tight_layout()
        plt.show()


def transform_number_under_10(x):
    if x < 10:
        x = '0' + str(x)
        return x
    else:
        x = str(x)
        return x
    
def get_population_data(list): 
    
    column_names = [i for i in range(0, 15)]
    population_df = pd.DataFrame(columns=column_names)
    
    print(f'Loading all files from ../data/input/population/: {list}')
    
    for filename in tqdm(list): 

        # print(f'{filename}')
        
        # extract year and half_year from filename
        temp = re.search(r'(\d{4})h(\d{2})', filename)
        year, half_year = temp.groups()

        # create file path for file import 
        file_path = f'../data/input/population/{filename}'

        # import data from 'T2' sheet - checked in Excel 
        raw_data = pd.read_excel(file_path, 'T2', header = None)

        # checked with excel the dataset before -> after dropping rows with Nan values the needed information are extracted 
        final_rows = raw_data.dropna().index.tolist()
        pop_data = raw_data.iloc[final_rows]
            
        # Add 2 columns year and half_year
        pop_data.insert(0, 'year', year)
        pop_data.insert(1, 'half_year', half_year)
        
        # counter = counter + len(pop_data)
            
        population_df = pd.concat([population_df, pop_data], ignore_index=True , axis=0)

        # Convert columns to integer 
        columns_to_transform = population_df.columns[0:17]
        population_df[columns_to_transform] = population_df[columns_to_transform].astype(int)
        # print(population_df.dtypes)
        
        # Transform format for LOR values - (1 -> 01, 2 -> 02 ...)
        for col in [0,1,2,3]:
            population_df[col] = population_df[col].apply(lambda x: transform_number_under_10(x))
        
        # drop woman and asyl data columns 
        # transform 0-3 to LOR Code
        # Rearrange table columns    
        population_df['lor'] = population_df[0] + population_df[1] + population_df[2] + population_df[3]
        population_df['key'] = population_df['lor'] + '-' + population_df['year'].astype(str) + '-' +  population_df['half_year'].astype(str)
    
    # Drop unnecessary columns 
    population_df = population_df.drop(columns = [0, 1, 2, 3, 13, 14])

    # Reorder and rename columns 
    population_df = population_df.reindex(['key', 'year', 'half_year', 'lor', 4, 5, 6, 7, 8, 9, 10, 11, 12], axis = 1)
    population_df.columns = ['key', 'year', 'half_year', 'lor', 'total_population', '-6', '6-15', '15-18', '18-27', '27-45', '45-55', '55-65', '65+']

    # Save the final DataFrame to a pickle file 
    population_df.to_pickle('../data/output/temp_analysis/total_population_dataset.pkl')
    
    print(f'Population data files where successfully loaded, combined and stored as a pickle file. \nPath: ../data/output/temp_analysis/total_population_dataset.pkl ')
    print('Dataframe preview:')
    return population_df