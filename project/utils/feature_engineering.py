import pandas as pd
import numpy as np

def preprocess_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses customer data for ML models.
    
    - Encodes 'Geography' into integers
      France -> 3, Germany -> 2, Others -> 1
    - Converts 'Gender' to binary (Male=1, Female=0)
    - Drops the original 'Geography' column
    
    Parameters:
        df (pd.DataFrame): Raw customer DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Encode Geography
    df['Geography'] = np.where(
        df['Geography'] == 'France', 3,
        np.where(df['Geography'] == 'Germany', 2, 1)
    )
    
    # Encode Gender
    # df['Gender'] = (df['Gender'] == 'Male').astype(int)
    df['Gender'] = encode_gender(df['Gender'])
    
    # Drop original Geography column
    # df = df.drop(columns=['Geography'], axis=1)
    
    return df



def encode_gender(x):
    """
    Universal gender encoder.
    Works for:
    - pandas Series
    - single value
    """

    # Case 1 → Pandas Series (training data)
    if isinstance(x, pd.Series):
        return (
            x.astype(str)
             .str.strip()
             .str.lower()
             .eq("male")
             .astype(int)
        )

    # Case 2 → Single value (prediction)
    return int(str(x).strip().lower() == "male")
