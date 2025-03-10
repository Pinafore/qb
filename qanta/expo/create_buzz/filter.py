import pandas as pd
import argparse

def filter_csv(input_file, output_file):
    # Load CSV file
    data = pd.read_csv(input_file, dtype=str)

    # Filter rows where "Answer" is not "correct"
    filtered_data = data[data["Answer"] != "Correct."]

    # Save the filtered data to a new CSV file
    filtered_data.to_csv(output_file, index=False)

    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    filter_csv("gpt_answers_2024.csv", "um...actually_2024.buzz.csv")
