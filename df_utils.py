import pandas as pd
from sklearn import preprocessing

def one_hot_encoding(df, column):
    df = df.copy()
    encoder = preprocessing.OneHotEncoder()
    onehotarray = encoder.fit_transform(df[[column]]).toarray()
    items = [f'{column}_{item}' for item in encoder.categories_[0]]
    df[items] = onehotarray
    df = df.drop(column, axis=1)
    return df

def fill_with_mean(col, df):
    mean_value = df[col].mean()
    df[col] = df[col].fillna(value=mean_value)

def fill_with_mode(col, df):
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(value=mode_value)

def prepare_df(df_train, df_test, max_unique = 0, skip_cols=None):
  df_train_copy = df_train.copy()
  df_test_copy = df_test.copy()
  for column in df_train_copy:
    if column in skip_cols:
      continue
    # print(column)
    column_type = df_train_copy[column].dtype
    check_nan = df_train_copy[column].isnull().values.any()
    # print(f'check_nan = {check_nan}')
    if column_type == 'object':
        # print(f'The column {column} contains string data')
        unique = df_train_copy[column].nunique()
        # print(f'unique = {unique}')
        if unique > max_unique:
          df_train_copy = df_train_copy.drop([column], axis=1)
          df_test_copy = df_test_copy.drop([column], axis=1)
          continue
        if check_nan:
          fill_with_mode(column, df_train_copy)
          fill_with_mode(column, df_test_copy)
        df_train_copy = one_hot_encoding(df_train_copy, column)
        df_test_copy = one_hot_encoding(df_test_copy, column)
    else:
        # print(f'The column {column} does not contain string data')
        if check_nan:
          fill_with_mean(column, df_train_copy)
          fill_with_mean(column, df_test_copy)
  return df_train_copy, df_test_copy

def drop_col_miss_val(df_train, df_test, perc):
  percent_missing = df_train.isnull().sum() * 100 / len(df_train)
  missing_value_df = pd.DataFrame({'column_name': df_train.columns,
                                  'percent_missing': percent_missing})
  missing_value_df = missing_value_df.loc[missing_value_df['percent_missing'] >= perc]
  cols = missing_value_df[['column_name']].values.squeeze(axis=1)
  # print(cols)
  df_train = df_train.drop(cols, axis=1, inplace=True)
  df_test = df_test.drop(cols, axis=1, inplace=True)  