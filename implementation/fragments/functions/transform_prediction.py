import pandas as pd
from joblib import load


# Function that transforms the dataset **for the prediction**
def transform_df_predict(df, original_name_dataset):

    # Loading the correspondent joblibs for the transformation
    drop_column_names = load(f'../joblibs/{original_name_dataset}/etl/drop_columns_names.joblib')
    encoder = load(f'../joblibs/{original_name_dataset}/etl/encoder.joblib')
    imp_mean = load(f'../joblibs/{original_name_dataset}/etl/imputation.joblib')
    df_mean_std = load(f'../joblibs/{original_name_dataset}/etl/mean_std.joblib')

    # Dropping the columns to drop
    df = df.drop(drop_column_names, axis=1
    