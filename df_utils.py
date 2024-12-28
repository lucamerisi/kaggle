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
    df[col] = df[col].fillna(df[col].mode()[0])