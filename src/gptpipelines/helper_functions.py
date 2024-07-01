def truncate(string, max_length):
    """
    Truncates a string to a maximum specified length, appending "..." if truncation occurs. The maximum length includes the appended '...'.

    If the string length exceeds `max_length`, it is truncated and ends with "...". 
    If the string length is less than or equal to `max_length`, it is returned unchanged.

    Parameters
    ----------
    string : str
        The string to potentially truncate.
    max_length : int
        The maximum allowed length of the string, including the ellipsis if truncation occurs.

    Returns
    -------
    str
        The truncated string with "..." appended if truncation occurred, 
        or the original string if no truncation was necessary.
    """

    formatted_string = string.replace("\\r\\n", " ")
    formatted_string = formatted_string.replace("\\n", " ")
    if max_length > 0 and len(string) > max_length:
        formatted_string = formatted_string[0:max_length-3] + "..."
        return formatted_string
    else:
        return formatted_string
    
def get_incomplete_entries(df, complete_feature):
    """
    Filters the input DataFrame to return rows where the value in the specified column is not 1.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to filter.
    complete_feature : str
        The column name to use for filtering. Rows with a value of 1 in this column are considered "complete" and are excluded.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the rows where the value in `complete_feature` is not 1, retaining the original indices.
    """
    
    # Filter the DataFrame based on the condition
    incomplete_df = df[df[complete_feature] != 1]
    
    return incomplete_df

def all_entries_are_true(dictionary):
    """
    Checks if all values in the provided dictionary are True.

    Parameters
    ----------
    dictionary : dict
        The dictionary to check, where the values are expected to be boolean.

    Returns
    -------
    bool
        True if all values in the dictionary are True, False otherwise.
    """

    for entry in dictionary:
        if dictionary[entry] is False:
            return False
    return True

def get_unique_columns_and_dtypes(dfs):
    """
    Takes a list of DataFrames and returns a list of unique column names and their data types.

    Parameters:
    dfs (list): List of pandas DataFrames.

    Returns:
    list: List of tuples, where each tuple contains a unique column name and its data type.
    """
    unique_columns_and_dtypes = {}
    
    for df in dfs:
        for col in df.columns:
            col_name = col.strip()
            col_dtype = df[col].dtype
            
            if col_name not in unique_columns_and_dtypes:
                unique_columns_and_dtypes[col_name] = col_dtype

    # Convert the dictionary to a list of tuples
    unique_columns_and_dtypes_list = [(col, dtype) for col, dtype in unique_columns_and_dtypes.items()]
    
    return unique_columns_and_dtypes_list