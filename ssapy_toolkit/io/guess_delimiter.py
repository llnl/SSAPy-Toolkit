import csv
from typing import List


def guess_csv_delimiter(csv_file_path: str, sample_size: int = 32768, delimiters: List[str] = [',', ';', '\t', '|', ' ']) -> str:
    """
    Guesses the delimiter used in a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        sample_size (int): The number of bytes to read from the file to guess the delimiter.
        delimiters (List[str]): The list of possible delimiters to test.

    Returns:
        str: The detected delimiter or an error message if unable to detect.
    """
    with open(csv_file_path, 'r') as csvfile:
        sample = csvfile.read(sample_size)  # Read a larger sample size
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
            print(f"Guessed {dialect.delimiter} delimiter.")
            return dialect.delimiter
        except csv.Error:
            return "Could not determine delimiter"
