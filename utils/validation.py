# Utility functions for validation and helpers
import pandas as pd
from typing import List

def validate_weather_columns(df: pd.DataFrame) -> bool:
    required = {'date', 'temp_c', 'precip_mm'}
    return required.issubset({c.lower() for c in df.columns})

def validate_sales_columns(df: pd.DataFrame) -> bool:
    required = {'region', 'sales', 'date'}
    return required.issubset({c.lower() for c in df.columns})

def validate_column_types(df: pd.DataFrame, col_types: dict) -> bool:
    for col, typ in col_types.items():
        if col in df.columns:
            if not pd.api.types.is_dtype_equal(df[col].dtype, typ):
                return False
    return True
