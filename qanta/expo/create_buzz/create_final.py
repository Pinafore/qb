import openai
from openai import OpenAI
import csv
import re
import uuid
import pandas as pd
from input_paraphaser import paraphaser
from query_generator import query_generator
from fact_checker import FactChecker
from answer_generator import AnswerGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI API key (replace 'your-api-key' with your actual key)
api_key = 'sk-onjTqMUfU5ZHzLdC440UYydenU74ZfRiLU985ROpw0BNQoEN'

paraphaser = paraphaser()
query_generator = query_generator()
answer_generator = AnswerGenerator()
fact_checker = FactChecker()

def check_fact(query, sentence, client):
    # Construct prompt
    prompt = "statement:\n" + sentence + "\n\nquestion:\n" + query
    
    fact_checker_response = fact_checker.create_chat(prompt, client)
    
    return fact_checker_response

def fact_checking_system(statement, client):
    '''
    This function is used to generate answer using our system
    :param statement: the full statement to be fact-checked
    :param client: the OpenAI client
    :return: answer , confidence
    '''
    paraphaser_response = paraphaser.create_chat("", statement, client)  # The full statement
    query_generator_response = query_generator.create_chat(paraphaser_response, client)
    query_list = [s for s in query_generator_response.split('\n') if s.strip()]

    with ThreadPoolExecutor() as executor:
        answer_list = list(executor.map(lambda query: check_fact(query, statement, client), query_list))

    final_prompt = 'statement:\n' + paraphaser_response + '\n\nkey points:\n'

    for i in range(len(answer_list)):
        final_prompt += query_list[i] + '\n' + answer_list[i] + '\n\n'

    final_response, confidence = answer_generator.create_chat(final_prompt, client)

    return final_response, confidence

error_count = 0
def ask_gpt_2(statement):
    global error_count
    client = OpenAI(base_url='https://api.openai-proxy.org/v1', api_key=api_key)
    
    try:
        answer, confidence = fact_checking_system(statement, client)
    except Exception as e:
        print(f"Error with fact_checking_system: {e}")
        answer, confidence = "Error", -100
        error_count += 1
    return answer, confidence

def write_to_csv(data, filename="gpt_answers_temp.csv"):
    """Writes the results to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Word Count", "Statement", "Answer", "Confidence Score", "Answered?"])

        for row in data:
            writer.writerow(row)

def process_statements_from_csv(input_csv):
    """Reads 'Um, actually...' statements with IDs from a CSV file, processes them, and writes results to a new CSV."""
    df = pd.read_csv(input_csv)

    # Check if required columns exist
    if "id" not in df.columns or "text" not in df.columns:
        print("Error: CSV file must contain 'id' and 'text' columns.")
        return

    statements = df.dropna(subset=["text"])  # Remove rows where 'text' is NaN
    results = []

    for _, row in statements.iterrows():
        statement_id = row["id"]
        statement = row["text"].strip()
        answer, confidence = ask_gpt_2(statement)
        print("Answer: ", answer)
        print("Confidence: ", confidence)
        word_count = len(statement.split())
        answered = "Yes" if confidence > -0.5 and answer.lower()[0:7] != 'correct' else "No"
        results.append([statement_id, word_count, statement, answer, confidence, answered])

    write_to_csv(results)
    print(f"Results written to 'gpt_answers.csv'.")
    print(error_count)

# Example Usage: Read from a CSV file
process_statements_from_csv("um...actually.csv")
