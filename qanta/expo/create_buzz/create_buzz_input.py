import openai
from openai import OpenAI
import csv
import re
import uuid
import textwrap
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

def split_statement(statement, max_chunk_size=30):
    """Splits a statement into sentences and further chunks long sentences."""
    sentences = re.split(r'(?<=[,.])\s+', statement)
    chunks = []

    for i, sentence in enumerate(sentences):
        words = sentence.split()

        if len(words) > max_chunk_size:
            sub_chunks = textwrap.wrap(sentence, width=max_chunk_size, break_long_words=False)
            for sub_chunk in sub_chunks:
                chunks.append((i + 1, sub_chunk.strip()))  # Assign same sentence number
        else:
            chunks.append((i + 1, sentence.strip()))

    return chunks

def check_fact(query, sentence, client):
    # 构造 prompt
    prompt = "statement:\n" + sentence + "\n\nquestion:\n" + query


    fact_checker_response = fact_checker.create_chat(prompt, client)

    return fact_checker_response


def fact_checking_system(context, sentence, client):
    '''
    This function is used to generate answer using our system
    :param context: the preceding context
    :param sentence: the sentence which we want it to be fact checked
    :param client: the OpenAI client
    :return: answer , confidence
    '''
    paraphaser_response = paraphaser.create_chat(context, sentence, client)  # a sentence
    query_generator_response = query_generator.create_chat(paraphaser_response,client)
    query = query_generator_response.split('\n')
    query_list = [s for s in query if s.strip()]

    with ThreadPoolExecutor() as executor:
        answer_list = list(executor.map(lambda query: check_fact(query, sentence, client), query_list))

    final_prompt = 'statement:\n' + paraphaser_response + '\n\nkey points:\n'

    for i in range(len(answer_list)):
        final_prompt += query_list[i] + '\n' + answer_list[i] + '\n\n'

    final_response, confidence = answer_generator.create_chat(final_prompt, client)

    return final_response, confidence

error_count = 0
def ask_gpt_2(paragraph):
    global error_count
    client = OpenAI(base_url='https://api.openai-proxy.org/v1',api_key=api_key)

    #split paragraph into context and sentence
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    if len(sentences) < 2:
        sentence = paragraph
        context = ""
    else:
        sentence = sentences[-1]
        context = ' '.join(sentences[:-1])
    #answer, confidence = fact_checking_system(context, sentence, client)

    try:
        answer, confidence = fact_checking_system(context, sentence, client)
    except Exception as e:
        print(f"Error with fact_checking_system: {e}")
        answer, confidence = "Error", -100
        error_count += 1
    return answer, confidence

def ask_gpt(context):
    """Sends an incrementally growing context to GPT-4o and retrieves a response with confidence score."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a fact-checking assistant."},
                      {"role": "user", "content": context}],
            temperature=0,
            max_tokens=200,
            top_p=0.95,
            logprobs=True  # Request log probabilities
        )

        answer = response['choices'][0]['message']['content']

        # Extract log probabilities if available
        log_probs = response['choices'][0].get('logprobs', {}).get('token_logprobs', [])
        # confidence1 = sum(log_probs) / len(log_probs)
        print("confidence", sum(log_probs), " and len ", len(log_probs))
        confidence = sum(log_probs) / len(log_probs) if log_probs else -100  # Default low confidence

        return answer, confidence

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error", -100  # Assign a very low confidence in case of errors


def write_to_csv(data, filename="gpt_answers_2024.csv"):
    """Writes the results to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Statement Number", "Word Count", "Sentence Number", "Chunk ID", "Chunk", "Answer",
                         "Confidence Score", "Answered?"])

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
        statement = row["text"]
        chunks = split_statement(statement)
        context = ""  # Accumulating context for the current statement

        for sentence_number, chunk in chunks:
            context += " " + chunk  # Merge previous chunks with the current one
            answer, confidence = ask_gpt_2(context.strip())
            print("Answer: ", answer)
            print("Confidence: ", confidence)
            word_count = len(context.split())
            chunk_uid = str(uuid.uuid4())[:8]  # Unique ID for each chunk
            if confidence > -0.5 and answer.lower()[0:7]!='correct':
                answered = "Yes"
            else:
                answered = "No"
            results.append(
                [statement_id, _, word_count, sentence_number, context, chunk_uid, answer, confidence, answered])

            # **Stop sending further chunks for this statement if GPT has confidently answered**
            if answered == "Yes":
                print(f"GPT answered confidently for statement ID {statement_id}. Moving to the next statement.")
                break  # Stop processing this statement and move to the next one

    write_to_csv(results)
    print(f"Results written to 'gpt_answers.csv'.")
    print(error_count)

# Example Usage: Read from a CSV file
process_statements_from_csv("um...actually_2024.csv")