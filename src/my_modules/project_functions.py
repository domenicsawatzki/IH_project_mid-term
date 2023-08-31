import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re # for regular expressions
from tqdm import tqdm # for progress bar
from pyproj import Proj, Transformer
from tqdm import tqdm
import os # for file system operations
import geopandas as gpd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from PyPDF3 import PdfFileReader # for reading PDF files
from tabula import read_pdf # for reading PDF files



# transform to snake_case, lower column names and strip
def clean_column_names(df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df is not a valid DataFrame.")
    
    # create copy
    clean_df = df.copy()
    
    #cleaning
    clean_df.columns= [col.strip() for col in clean_df.columns]
    clean_df.columns= [col.lower() for col in clean_df.columns]
    clean_df.columns= [col.replace(' ', '_') for col in clean_df.columns]
    
    return clean_df

def change_to_int(x):
    if isinstance(x, int):
        return x
    else: 
        if x == '0':
            x = int(0)
        elif x == '1':
            x = int(1)
        else:
            x = int(0)   

        return x 

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
    
def add_zero_at_beginning(x):
    if len(x) == 7:
        return '0' + x
    return x
    
def get_population_data(list): 
    
    column_names = [i for i in range(0, 15)]
    df = pd.DataFrame(columns=column_names)
    
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
        temp_df = raw_data.iloc[final_rows]
            
        # Add 2 columns year and half_year
        temp_df.insert(0, 'year', year)
        temp_df.insert(1, 'half_year', half_year)
        
        # counter = counter + len(temp_df)
            
        df = pd.concat([df, temp_df], ignore_index=True , axis=0)

        # Convert columns to integer 
        columns_to_transform = df.columns[0:17]
        df[columns_to_transform] = df[columns_to_transform].astype(int)
        # print(df.dtypes)
        
        # Transform format for LOR values - (1 -> 01, 2 -> 02 ...)
        for col in [0,1,2,3]:
            df[col] = df[col].apply(lambda x: transform_number_under_10(x))
        
        # drop woman and asyl data columns 
        # transform 0-3 to LOR Code
        # Rearrange table columns    
        df['lor'] = df[0] + df[1] + df[2] + df[3]
        df['key'] = df['lor'] + '-' + df['year'].astype(str) + '-' +  df['half_year'].astype(str)
    
    # Drop unnecessary columns 
    df = df.drop(columns = [0, 1, 2, 3, 13, 14])

    # Reorder and rename columns 
    df = df.reindex(['key', 'year', 'half_year', 'lor', 4, 5, 6, 7, 8, 9, 10, 11, 12], axis = 1)
    df.columns = ['key', 'year', 'half_year', 'lor', 'total_population', '-6', '6-15', '15-18', '18-27', '27-45', '45-55', '55-65', '65+']

    # Save the final DataFrame to a pickle file 
    df.to_pickle('../data/output/temp_analysis/total_population_dataset.pkl')
    
    print(f'Population data files where successfully loaded, combined and stored as a pickle file. \nPath: ../data/output/temp_analysis/total_population_dataset.pkl ')
    print('Dataframe preview:')
    display(df.head(5))
    return df

#TODO: create a for loop going through all files in../data/input/population
def get_accident_data():
    list_accident_files = os.listdir('../data/input/road_accidents/')
    list_accident_files 
    
    print(f'Loading all files from ../data/input/road_accidents/: {list_accident_files}')
    accidents18 = pd.read_csv('../data/input/road_accidents/berlin_road_accidents_2018.csv', sep=';', encoding = 'latin1')
    accidents19 = pd.read_csv('../data/input/road_accidents/berlin_road_accidents_2019.csv', sep=';', encoding = 'latin1')
    accidents20 = pd.read_csv('../data/input/road_accidents/berlin_road_accidents_2020.csv', sep=';', encoding = 'latin1')
    accidents21 = pd.read_csv('../data/input/road_accidents/berlin_road_accidents_2021.csv', sep=';', encoding = 'latin1')

    accidents18 = clean_column_names(accidents18)
    accidents19 = clean_column_names(accidents19)
    accidents20 = clean_column_names(accidents20)
    accidents21 = clean_column_names(accidents21)
    
    # add empty lor_ab_2021 column on accidents19
    #TODO: cleaning function
    accidents18.rename(columns={'strzustand':'ustrzustand'}, inplace=True)
    accidents18.rename(columns={'istsonstig':'istsonstige'}, inplace=True)
    accidents19.insert(5, 'lor_ab_2021', '')
    accidents20.insert(4, 'strasse', '')
    accidents21.insert(3, 'lor', '')
    accidents21.insert(4, 'strasse', '')
    
    final_df = pd.concat([accidents18, accidents19, accidents20, accidents21], axis=0)
    final_df.shape

    final_df = final_df.drop(columns=['land'])
    final_df.dtypes
    
        
    final_df = final_df.fillna('0')
    final_df = final_df.replace("",'0')

    final_df['lor'] = final_df['lor'].astype(float)
    final_df['lor'] = final_df['lor'].astype('Int64')

    final_df['lor_ab_2021'] = final_df['lor_ab_2021'].astype(float)
    final_df['lor_ab_2021'] = final_df['lor_ab_2021'].astype('Int64')

    final_df['lor_ab_2021'] = final_df['lor_ab_2021'].astype(float)
    final_df['lor_ab_2021'] = final_df['lor_ab_2021'].astype('Int64')


    final_df['istsonstige'] = final_df['istsonstige'].apply(lambda x: change_to_int(x))
    final_df['ustrzustand'] = final_df['ustrzustand'].apply(lambda x: change_to_int(x))

    # transform 
    final_df['linrefx'] = final_df['linrefx'].str.replace(",",'.').astype(float)
    final_df['linrefy'] = final_df['linrefy'].str.replace(",",'.').astype(float)
    final_df['xgcswgs84'] = final_df['xgcswgs84'].str.replace(",",'.').astype(float)
    final_df['ygcswgs84'] = final_df['ygcswgs84'].str.replace(",",'.').astype(float)
    
    
    final_df = final_df.drop(columns=['bez'])
    
    # rename columns
    final_df.columns = ['object_id', 'old_lor','street_default', 'lor', 'year', 'month',
        'hour', 'weekday', 'ac_category', 'ac_type', 'ac_type2', 'ac_light',
        'is_bicycle', 'is_car', 'is_pedestrian', 'is_motorcycle', 'is_truck', 'is_other',
        'street_condition', 'linrefx', 'linrefy', 'xgcswgs84', 'ygcswgs84']
    
    key = final_df['object_id'].astype(str) + '-' + final_df['year'].astype(str) + '-' + final_df['lor'].astype(str)
    final_df.insert(0, 'key', key)
    
    #trasnform to bool
    bool_list = ['is_bicycle', 'is_car', 'is_pedestrian', 'is_motorcycle', 'is_truck', 'is_other']
    
    
    tqdm.pandas()
    # ETRS89/UTM Zone 32N
    utm = Proj('epsg:25832')

    # WGS84
    wgs = Proj('epsg:4326')
    transformer = Transformer.from_crs('epsg:25832', 'epsg:4326')

    lat, lon = transformer.transform(final_df['linrefx'].tolist(), final_df['linrefy'].tolist())

    final_df['latitude'] = lat
    final_df['longitude'] = lon
    
    for l in bool_list:
        final_df[l] = final_df[l].astype(bool)
    
    # transform to object
    object_list = ['weekday', 'ac_category', 'ac_type', 'ac_type2', 'ac_light', 'street_condition']

    for l in object_list:
        final_df[l] = final_df[l].astype('object')
    
    # transform to string
    final_df['lor'] = final_df['lor'].astype(str)
    final_df['old_lor'] = final_df['old_lor'].astype(str)
    
    final_df['lor'] = final_df['lor'].apply(add_zero_at_beginning)
    final_df['old_lor'] = final_df['old_lor'].apply(add_zero_at_beginning)
    
    final_df['lor_3'] = final_df['lor'].str[:-2]
    final_df['lor_2'] = final_df['lor'].str[:-4]
    final_df['lor_1'] = final_df['lor'].str[:-6]
    

    
    final_df.to_pickle('../data/output/temp_analysis/accident_dataset.pkl')
    print(f'Accident data files where successfully loaded, combined and stored as a pickle file. \nPath: ../data/output/temp_analysis/accident_dataset.pkl ')
    print('Dataframe preview:')
    display(final_df.head(5))
    return final_df



def get_adress_data_by_year(year):
    path = f'../data/input/berlin_adresses/{year}'
    
    list_of_files = os.listdir(path)
    list_of_files
    
    length = len(list_of_files)
    counter = 0 
    
    column_names = [i for i in range(0, 14)]
    # print(len(column_names))
    final_df = pd.DataFrame(columns = column_names)
    
    print(f'Extracting data from files in {path}')
    
    for file_name in (list_of_files):
        path = f'../data/input/berlin_adresses/{year}/{file_name}'
        df, name, counter = extract_all_pages_from_pdf(path, file_name, counter, length)
        
        # df.to_csv(f'../data/output/temp_adress_data/{name}-{year}.csv', index = False)
        
        final_df = pd.concat([final_df, df], ignore_index=True , axis=0)
    
    final_df.to_pickle('../data/output/temp_analysis/adress_dataset.pkl')  
        # display(dataset.head(5))
        # print(dataset.shape)
    return final_df

def extract_all_pages_from_pdf(path, file_name, counter, lenght):
    with open(path, 'rb') as f:
        # open path as binary / rb = read binary -> used for pdf files
        counter += 1        
        pdf = PdfFileReader(f)
        last_page = pdf.getNumPages()
        last_page += 1

        # defining page range for import 
        [i for i in range(0, 15)]
        page_range = [i for i in range(5, last_page)]
        # print(page_range)

        # prepare column name filler
        column_names = [i for i in range(0, 14)]
        # print(column_names)

        # create empty dataframe
        adress_data = pd.DataFrame(columns = column_names)
        # display(adress_data)
        
        # extract district information 
        #TODO: nochmal prüfen und vielleicht optimieren
        district = [re.findall(r'(?<=adr_)(.*?)(?=_\d{4}\.pdf)', file_name)]
        print(f'file {counter}/{lenght} - {district[0][0]}')
        for page in tqdm(page_range):
            
            import_data = read_pdf(path, pages = page, encoding = 'ISO-8859-1', stream = True, area = [175, 33, 783, 520], guess = True, pandas_options={'header': None})
            table_df = import_data[0]

            
            table_df = import_data[0]
            columns_len = len(table_df.columns)

            if columns_len < 14:
                if table_df.iloc[:, 3].dtype in ['int64', 'int32', 'float64', 'float32']:
                    table_df.insert(3, 'm1', np.nan)

                if table_df.iloc[:, 6].dtype in ['int64', 'int32', 'float64', 'float32']:
                    table_df.insert(5, 'm2', np.nan)
                
                
                column_length = len(table_df.columns)

                
                if column_length < 14:
                    table_df['m3'] = np.nan
                    # display(table_df)
                
                
                table_df.columns = column_names
                table_df['14'] = str(district[0][0])
                
                adress_data = pd.concat([adress_data, table_df], ignore_index=True , axis=0)

        return adress_data, district[0][0], counter
    
def import_geo_data():
    list_accident_files = os.listdir('../data/input/geo_data/')
    list_accident_files 
    print(f'Loading all files from ../data/input/geo_data/: {list_accident_files}')
    # Planungsebene 
    path = '../data/input/geo_data/lor_planungsraeume_2021.geojson'
    with open(path, 'r', encoding='utf-8') as f:
        gdf_plan = gpd.read_file(f)
    
    gdf_plan = gdf_plan.drop(columns =['BEZ','STAND'] )
    gdf_plan['GROESSE_M2'] = gdf_plan['GROESSE_M2'].apply(lambda x: round(x,2))

    gdf_plan.columns = ['lor_4', 'lor4_name', 'area_in_sqm_lor4', 'geometry_lor4']

    
    # bezirksebene
    path = '../data/input/geo_data/lor_bezirksregionen_2021.geojson'
    with open(path, 'r', encoding='utf-8') as f:
        gdf_region = gpd.read_file(f)

    gdf_region = gdf_region.drop(columns =['BEZ','STAND', 'GROESSE_m2'])
    gdf_region.columns = ['lor_3', 'lor_3_name','geometry_lor3']


    # prognoseebene
    path = '../data/input/geo_data/lor_prognoseraeume_2021.geojson'
    with open(path, 'r', encoding='utf-8') as f:
        gdf_prognose = gpd.read_file(f)
    
    gdf_prognose
    gdf_prognose = gdf_prognose.drop(columns =['BEZ','STAND', 'GROESSE_M2'])
    gdf_prognose.columns = ['lor_2', 'lor_2_name','geometry_lor2']
    
    # bezirk
    path = '../data/input/geo_data/lor_ortsteile.geojson'
    with open(path, 'r', encoding='utf-8') as f:
        gdf_ort = gpd.read_file(f)

    gdf_ort = gdf_ort.drop(columns=[ 'gml_id', 'spatial_alias','spatial_type', 'FLAECHE_HA',  'OTEIL', 'geometry']) 
    gdf_ort.columns = ['lor_1', 'lor_1_name']

    gdf_ort['lor_1']= gdf_ort['lor_1'].str[:2] 
    gdf_ort = gdf_ort.drop_duplicates()

    
    gdf_total = pd.merge(gdf_plan, gdf_region, left_on = gdf_plan['lor_4'].str[:6], right_on = 'lor_3', how = 'left')
    gdf_total = pd.merge(gdf_total, gdf_prognose, left_on = gdf_plan['lor_4'].str[:4], right_on = 'lor_2', how = 'left')
    gdf_total = pd.merge(gdf_total, gdf_ort, left_on = gdf_plan['lor_4'].str[:2], right_on = 'lor_1', how = 'left')
    
    gdf_total.to_pickle('../data/output/temp_analysis/gdf_data.pkl')
    print(f'Geo data files where successfully loaded, combined and stored as a pickle file. \nPath: ../data/output/temp_analysis/gdf_data.pkl ')
    print('Dataframe preview:')
    display(gdf_total.head(5))
    
    return gdf_total

def get_distance(row):
    from geopy.distance import geodesic as GD
    latitude = row['latitude_x']
    longitude = row['longitude_x']
    point = (latitude, longitude)
    city_center = (52.51916459, 13.405665044)
    distance = GD(point, city_center).km
    return distance 



def data_wrangling(accident_df, geo_df, population_df):
    accident_df = accident_df.drop(columns = ['street_default', 'is_motorcycle', 'is_truck','is_other', 'linrefx', 'linrefy', 'xgcswgs84','ygcswgs84', 'ac_type2', 'street_condition', 'ac_light', 'object_id'])
    
    accident_df['ac_category'] = accident_df['ac_category'].replace({1:'deadly', 2:'seriously_injured', 3:'light_injured' })
    accident_df['ac_type'] = np.where(accident_df['ac_type'].isin([1, 4, 5, 6, 7, 8, 9, 0]), 'other', 'turn_in')
    print('Start finalizing accident dataset.')
    print('Calculating distances to city center for each row:')
    accident_df['distance_to_CC'] = accident_df.progress_apply(get_distance, axis =1)
    
    print(f'Final accident datafile is saved as csv and xslx. \nPath: ../data/output/tableau\n')
    accident_df.to_csv('../data/output/Tableau/final_accident_data.csv', index=False)
    accident_df.to_excel('../data/output/Tableau/final_accident_data.xlsx', index=False)   

    print('Start combining geo and population data and create a summary table.')
    print('Calculating distances to city center for each row:')
    
    population_df = population_df.loc[:,['lor', 'total_population']]
    summary_table = pd.merge(population_df, geo_df, left_on = 'lor', right_on = 'lor_4', how = 'left')
    
    cycle_df = accident_df.loc[accident_df['is_bicycle'] == True]
    
    
    temp = cycle_df.groupby(['lor_x']).agg({'key':pd.Series.nunique, 'distance_to_CC':np.mean}).reset_index()
    temp.columns = ['lor', 'total_accident_per_lor', 'avg_distance_to_CC_per_lor']
    temp['total_accident_per_lor'] = temp['total_accident_per_lor'].astype(int)

    summary_table = pd.merge(summary_table, temp, on = 'lor', how = 'left')
    summary_table['population_density_per_lor_in_sqkm'] = round(summary_table['total_population'] / summary_table['area_in_sqm_lor4'] * 1000000,0)
    summary_table['total_accident_per_lor'].fillna(0, inplace=True)
    
    summary_table.to_csv('../data/output/Tableau/final_summary_table.csv', index=False)
    summary_table.to_excel('../data/output/Tableau/final_summary_table.xlsx', index=False)
    print(f'Final summary table is saved as csv and xslx. \nPath: ../data/output/tableau\n')
    return accident_df, summary_table

def predict_LOR_values_based_on_lat_lon(raw_df):
    
    df = raw_df[['key','lor', 'latitude', 'longitude']]

    df[['lor']] = df[['lor']].astype(str)

    df_model = df.loc[df['lor'] != '0']
    df_predict = df.loc[df['lor'] == '0']

    X = df_model.drop(columns = ['lor','key'])
    y = df_model['lor']
    
    X_predict = df_predict.drop(columns = ['lor','key'])

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    

    KNN = KNeighborsClassifier(n_neighbors=2, p=1, weights = 'distance')
    KNN.fit(X_train, y_train)
    
    score = KNN.score(X_test, y_test)
    print(f'R2 Testscore: {score}')

    score_train = KNN.score(X_train, y_train)
    print(f'R2 Train: {score_train}')
    
    y_predicted = KNN.predict(X_predict)
    df_predict['lor'] = y_predicted
    df_predict.drop(columns = ['latitude', 'longitude'])
    
    final_df = pd.merge(raw_df, df_predict, on = 'key', how = 'left')
    final_df.loc[final_df['lor_x'] == '0', 'lor_x'] = final_df['lor_y']
    final_df.drop(columns = ['latitude_y', 'longitude_y', 'lor_y'], inplace = True)
    

    
    return final_df