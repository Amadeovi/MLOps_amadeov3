"""Prepare customer data for churn prediction."""


def preprocess_data(df):
    """Preprocess the dataframe to be ready for training.

    Args:
        df: pd.DataFrame
    """
    df.columns = df.columns.str.lower().str.replace({"  ": "_", " ": "_"})

    return df
