import pandas as pd


def split_count(x):
    """
    Split comma-separated values in a pandas Series and count occurrences.
    
    Takes a pandas Series where each entry contains comma-separated values,
    splits them into individual items, and returns a DataFrame with counts
    of each unique item, sorted by count in descending order.
    
    Parameters
    - x : pd.Series
        A pandas Series containing comma-separated string values.
        NaN/None values are automatically skipped.
    
    Returns
    - pd.DataFrame
        A DataFrame with unique items as index and a 'count' column showing
        the frequency of each item. Sorted by count in descending order.
    """
    assert isinstance(x, pd.Series), \
        f"Input must be a pandas Series, got {type(x).__name__}"
    assert len(x) > 0, \
        "Input Series must not be empty"
    all_items = []
    
    for idx, entry in enumerate(x):
        if pd.isna(entry):
            continue
        assert isinstance(entry, str), \
            f"Series entry at index {idx} must be a string, got {type(entry).__name__}: {repr(entry)}"
        items = [item.strip() for item in entry.split(',') if item.strip()]
        all_items.extend(items)
    assert len(all_items) > 0, f"No valid comma-separated values found in the Series"
    counts = pd.Series(all_items).value_counts()
    result = pd.DataFrame({'count': counts})
    return result