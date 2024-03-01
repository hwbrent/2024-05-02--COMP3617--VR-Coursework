import os
import math

import pandas as pd


DATASET_NAME = "IMUData.csv"

this_dir = os.path.dirname(__file__)
parent_dir = os.path.join(this_dir, os.pardir)
dataset_path = os.path.join(parent_dir, DATASET_NAME)
dataset_path = os.path.abspath(dataset_path)


def import_dataset() -> pd.core.frame.DataFrame:
    """
    --- Problem 2 Question 1 ---

    This function loads the `IMUData` CSV file and returns it as a pandas
    `DataFrame`
    """
    ds = pd.read_csv(dataset_path)

    # For some reason, there is rogue whitespace on one or both sides of
    # each column name. Let's fix that:
    ds.columns = ds.columns.map(lambda col_name: col_name.strip())

    return ds


def convert_rotational_rate(dataset: pd.core.frame.DataFrame) -> None:
    """
    --- Problem 2 Question 1 ---

    Given the dataset obtained by `import_dataset`, this function converts
    the rotational rate (tri-axial velocity in deg/s) to radians/sec
    in-place
    """

    axes = ("X", "Y", "Z")
    for axis in axes:
        # Get the name of the column
        col = "gyroscope." + axis  # e.g. gyroscope.X

        # Convert each value in the column from degrees to radians
        dataset[col] = dataset[col].apply(lambda value: math.radians(value))


def get_dataset() -> pd.core.frame.DataFrame:
    """
    --- Problem 2 Question 1 ---

    This function returns the dataset with the rotational rate conversion
    applied to it
    """
    dataset = import_dataset()
    convert_rotational_rate(dataset)
    return dataset
