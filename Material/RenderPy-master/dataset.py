import os
import math

import pandas as pd
from pandas.core.frame import DataFrame


this_dir = os.path.dirname(__file__)
parent_dir = os.path.join(this_dir, os.pardir)


class Dataset:
    FILE_NAME = "IMUData.csv"

    def __init__(self) -> None:
        # Automatically load the dataset, and convert the rotational rate
        self.df = self.import_from_csv()
        self.convert_rotational_rate()

    def get_path(self) -> str:
        """
        This function returns the absolute filepath of the CSV file containing
        the dataset
        """
        path = os.path.join(parent_dir, self.FILE_NAME)
        abs_path = os.path.abspath(path)
        return abs_path

    def import_from_csv(self) -> DataFrame:
        """
        --- Problem 2 Question 1 ---

        This function loads the `IMUData` CSV file and returns it as a pandas
        `DataFrame`
        """
        df = pd.read_csv(self.get_path())

        # For some reason, there is rogue whitespace on one or both sides of
        # each column name. Let's fix that:
        df.columns = df.columns.map(lambda col_name: col_name.strip())

        return df

    def convert_rotational_rate(self) -> None:
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
            self.df[col] = self.df[col].apply(lambda value: math.radians(value))

    def __getitem__(self, key):
        return self.df[key]
