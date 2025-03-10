import pandas as pd
import csv

def process_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Convert 'Answered?' column to boolean values
    df['final'] = df['Answered?'].apply(lambda x: True if x == 'Yes' else False)
    
    # Create the output DataFrame with required transformations
    df_output = pd.DataFrame({
        'word': df['Word Count'],
        'sentence': df['Sentence Number'],
        'question': df['ID'],
        'page': df['Answer'],
        'evidence': df.apply(lambda row: {'confidence': row['Confidence Score'], 'final': row['final']}, axis=1),
        'weight': df['Confidence Score'],
        'final': df['final']
    })
    
    # Save to output CSV
    df_output.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
   
# Example usage
input_csv_file = "gpt_answers.csv"  # Replace with actual input CSV file path
output_csv_file = "um...actually.buzz.csv"  # Replace with desired output CSV file path
process_csv(input_csv_file, output_csv_file)
