import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    
def customerDataCleanAndPrepareForModel(df):
    import numpy as np
    df.columns= [col.lower() for col in df.columns]
    df.columns= [col.replace(' ', '_') for col in df.columns]
    df.rename(columns={'st':'state'}, inplace=True)
    df.rename(columns={'monthly_premium_auto':'monthly_premium_costs'}, inplace=True)
    df.drop(columns=['customer', 'effective_to_date'], axis=1)

    
    y = df['total_claim_amount']
    X = df.drop(['total_claim_amount'], axis=1)

    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)
    
    from sklearn.preprocessing import MinMaxScaler # do not use the function Normalise() - it does something entirely different
    MinMaxtransformer = MinMaxScaler().fit(X_num)
    
    X_normalized = MinMaxtransformer.transform(X_num)
    X_normalized = pd.DataFrame(X_normalized,columns=X_num.columns)

    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_cat)
    encoded_X_cat = encoder.transform(X_cat)
    # Convert the encoded data to a pandas DataFrame
    encoded_X_cat = pd.DataFrame(encoded_X_cat.toarray(), columns=encoder.get_feature_names_out(X_cat.columns))
    
    final_X = pd.concat([X_normalized ,encoded_X_cat], axis=1)

    return final_X, y

def clean_column_names(df):
    df.columns= [col.lower() for col in df.columns]
    df.columns= [col.replace(' ', '_') for col in df.columns]
    df.rename(columns={'st':'state'}, inplace=True)
    df.rename(columns={'monthly_premium_auto':'monthly_premium_costs'}, inplace=True)

    
    return df

def categorize_variables(df):
    continuous_df = pd.DataFrame()
    discrete_df = pd.DataFrame()

    for column in df.columns:
        unique_values = df[column].nunique()

        if unique_values > 10:
            continuous_df[column] = df[column]
        else:
            discrete_df[column] = df[column]

    return continuous_df, discrete_df

def plot_discrete_var(df):
    for col in df.columns:
        plt.figure(figsize=(5,5))
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
        sns.histplot(df[col], bins = bins, kde = True)
        plt.title(f"Histogram for {col}")
        
        # Boxplot
        plt.subplot(1,2,2)
        sns.boxplot(df[col])
        plt.title(f"Boxplot for {col}")
        
        plt.tight_layout()
        plt.show()
        

def define_upper_limit(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    upper_limit = Q3 + (IQR * 1.5)
    print(f"Upper limit for {col} is {upper_limit.round(2)}.")
    return upper_limit.round(2)


def check_unique_values(df):
    columns = df.columns

    for col in columns:
        print(
f"""
Unique Values for column: **{col}** -> Number of unique values: **{df[col].nunique()}**
{df[col].unique()}
Valuecounts: 
{df[col].value_counts()}
Dtype: {df[col].dtype}

""")