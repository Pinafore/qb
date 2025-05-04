import openai
from openai import OpenAI
import csv
import re
import uuid
import textwrap
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
from input_paraphaser import paraphaser
from query_generator import query_generator
from fact_checker import FactChecker
from answer_generator import AnswerGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI API key (replace 'your-api-key' with your actual key)
api_key = 'API key'

paraphaser = paraphaser()
query_generator = query_generator()
answer_generator = AnswerGenerator()
fact_checker = FactChecker()

def split_statement(statement, max_chunk_size=100):
    raw_chunks = statement.split(",")
    chunks = []
    for part in raw_chunks:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) > max_chunk_size:
            sub_chunks = textwrap.wrap(part, width=max_chunk_size, break_long_words=False)
            chunks.extend(sub.strip() for sub in sub_chunks)
        else:
            chunks.append(part)
    return chunks

def check_fact(query, sentence, client):
    prompt = f"statement:\n{sentence}\n\nquestion:\n{query}"
    return fact_checker.create_chat(prompt, client)

def fact_checking_system(context, sentence, client):
    paraphrased = paraphaser.create_chat(context, sentence, client)
    queries = query_generator.create_chat(paraphrased, client).split('\n')
    query_list = [q for q in queries if q.strip()]

    with ThreadPoolExecutor() as executor:
        answers = list(executor.map(lambda q: check_fact(q, sentence, client), query_list))

    final_prompt = f'statement:\n{paraphrased}\n\nkey points:\n'
    for q, a in zip(query_list, answers):
        final_prompt += f"{q}\n{a}\n\n"

    final_answer, confidence = answer_generator.create_chat(final_prompt, client)
    return final_answer, confidence

def ask_gpt_2(paragraph):
    client = OpenAI(base_url='https://api.openai.com/v1', api_key=api_key)
    sentences = re.split(r'(?<=[,])\s+', paragraph.strip())
    if len(sentences) < 2:
        sentence, context = paragraph, ""
    else:
        sentence = sentences[-1]
        context = ' '.join(sentences[:-1])

    try:
        return fact_checking_system(context, sentence, client)
    except Exception as e:
        print(f"Error with fact_checking_system: {e}")
        return "Error", -100

def write_to_csv(data, filename="gpt_answers_final.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Statement Number", "Word Count", "Sentence Number", "Chunk ID", "Chunk", "Answer", "Confidence Score", "Answered?"])
        writer.writerows(data)

def process_statements_from_csv(input_csv):
    df = pd.read_csv(input_csv)

    if not {'id', 'text', 'sent'}.issubset(df.columns):
        print("CSV must contain 'id', 'text', and 'sent' columns.")
        return

    results = []
    grouped = df.dropna(subset=["text"]).groupby("id")

    for statement_id, group in grouped:
        context_chunks = []
        answered_flag = False
        sent_word_count_map = {}  # Track cumulative word count for each sentence number

        for idx, row in group.iterrows():
            if answered_flag:
                break

            sentence_number = row["sent"]
            current_text = row["text"]
            chunks = split_statement(current_text)

            for chunk in chunks:
                if answered_flag:
                    break

                context_chunks.append(chunk)
                cumulative_context = ", ".join(context_chunks)
                chunk_word_count = len(chunk.split())

                # Update cumulative word count for this `sent`
                if sentence_number not in sent_word_count_map:
                    sent_word_count_map[sentence_number] = 0
                sent_word_count_map[sentence_number] += chunk_word_count
                total_word_count_for_sent = sent_word_count_map[sentence_number]

                print(f"\n[ID {statement_id}] Sending to GPT:", cumulative_context)
                answer, confidence = ask_gpt_2(cumulative_context)
                print("Answer:", answer)

                chunk_uid = str(uuid.uuid4())[:8]
                answered = "Yes" if confidence > -0.6 and not answer.lower().startswith("correct") else "No"

                results.append([
                    statement_id,
                    idx,
                    total_word_count_for_sent,
                    sentence_number,
                    chunk_uid,
                    cumulative_context,
                    answer,
                    confidence,
                    answered
                ])

                if answered == "Yes":
                    print(f"✅ Confident answer for ID {statement_id}. Moving on.")
                    answered_flag = True
                    break

    write_to_csv(results)
    print("All results saved to 'gpt_answers_final.csv'.")

# Example Usage:
process_statements_from_csv("um...actually.csv")
