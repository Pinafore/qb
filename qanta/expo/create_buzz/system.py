import requests
import random
from openai import OpenAI
import time
import ast
from jsonl_produce import read_jsonl, write_jsonl
from input_paraphaser import paraphaser
from query_generator import query_generator
from fact_checker import FactChecker
from answer_generator import AnswerGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed

input_filepath = ''
output_filepath = ''

paraphaser = paraphaser()
query_generator = query_generator()
answer_generator = AnswerGenerator()
fact_checker = FactChecker()


times = []
def traverse_paragraph(paragraph):
    import re


    sentences = re.split(r'([.!?])', paragraph)

    result = []
    current_text = ""


    for i in range(0, len(sentences) - 1, 2):  # 步长为2，保证每次处理一个句子和它的标点符号
        sentence = sentences[i] + sentences[i + 1]  # 句子和标点符号组合
        current_text += sentence  # 累积内容
        result.append(current_text)  # 保存当前的内容

    return result

def check_fact(query, sentence):
    # 构造 prompt
    prompt = "statement:\n" + sentence + "\n\nquestion:\n" + query

    # 创建消息并发送请求
    fact_checker_messages = fact_checker.build_messages(prompt)
    fact_checker_response = fact_checker.send_request(fact_checker_messages)

    return fact_checker_response

def produce_sentence(id, sentence,correction=None, correction2=None):
    data = []

    sentences = traverse_paragraph(sentence)
    #print(sentences)
    num = 0
    for sentence in sentences:
        num +=1
        print("\ncurrent input:")
        print(sentence)
        line = {}
        line['id'] = id
        line['input_id'] = num
        line['question'] = sentence
        start_time = time.time()
        paraphaser_messages = paraphaser.build_messages(sentence)
        paraphaser_response = paraphaser.send_request(paraphaser_messages)# a sentence
        #print("\nparaphaser response:")
        #print(paraphaser_response, end='\n')
        query_generator_messages = query_generator.build_messages(paraphaser_response)
        query_generator_response = query_generator.send_request(query_generator_messages)
        #print("\nquery list:\n")
        #print(query_generator_response, end='\n')
        query_list = ast.literal_eval(query_generator_response)

        with ThreadPoolExecutor() as executor:
            # 使用 map 保持顺序
            answer_list = list(executor.map(lambda query: check_fact(query, sentence), query_list))
        #print("\nanswers:\n")
        #print(answer_list)


        final_prompt = 'statement:\n' + paraphaser_response  + '\n\nkey points:\n'
        for i in range(len(answer_list)):
            final_prompt += query_list[i]+'\n'+ answer_list[i] + '\n\n'
        #print('\nfinal_prompt:')
        #print(final_prompt)
        final_messages = answer_generator.build_messages(final_prompt)
        final_response = answer_generator.send_request(final_messages)
        end_time = time.time()
        print('\nfinal response:\n' + final_response)
        line['answer'] = final_response
        print('gold_answer:\n' + correction+'\n' + str(correction2))
        #line['result'] = input('Enter result:')
        data.append(line)
        times.append(end_time-start_time)
        a = input("Continue? (y/n): ")
        if a.lower() == 'n':
            break

    return data

def extract_last_sentence(paragraph):
    """
    Extract the last sentence from a paragraph and return the last sentence and its context.

    Parameters:
        paragraph (str): The input paragraph.

    Returns:
        tuple: A tuple containing the context (str) and the last sentence (str).
    """
    import re

    # Split the paragraph into sentences using regex to handle different punctuation marks.
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())

    if len(sentences) > 1:
        context = ' '.join(sentences[:-1])
        last_sentence = sentences[-1]
    else:
        context = ''
        last_sentence = sentences[0]

    return context, last_sentence
def produce_input(sentence):
    print("\ncurrent input:")
    print(sentence)
    line = {}
    line['id'] = id
    line['input_id'] = 0
    line['question'] = sentence
    start_time = time.time()
    context, last_sentence = extract_last_sentence(sentence)
    paraphaser_messages = paraphaser.build_messages(context, last_sentence)
    paraphaser_response = paraphaser.send_request(paraphaser_messages)  # a sentence
    print("\nparaphaser response:")
    print(paraphaser_response, end='\n')
    query_generator_messages = query_generator.build_messages(paraphaser_response)
    query_generator_response = query_generator.send_request(query_generator_messages)
    print("\nquery list:\n")
    print(query_generator_response, end='\n')
    query_list = ast.literal_eval(query_generator_response)

    with ThreadPoolExecutor() as executor:
        # 使用 map 保持顺序
        answer_list = list(executor.map(lambda query: check_fact(query, sentence), query_list))
    #print("\nanswers:\n")
    # print(answer_list)

    final_prompt = 'statement:\n' + paraphaser_response + '\n\nkey points:\n'
    for i in range(len(answer_list)):
        final_prompt += query_list[i] + '\n' + answer_list[i] + '\n\n'
    print('\nfinal_prompt:')
    print(final_prompt)
    final_messages = answer_generator.build_messages(final_prompt)
    final_response = answer_generator.send_request(final_messages)
    end_time = time.time()
    print('\nfinal response:\n' + final_response)
    line['answer'] = final_response
    #print('gold_answer:\n' + correction + '\n' + str(correction2))
    # line['result'] = input('Enter result:')
    data.append(line)
    return end_time-start_time

def produce_input_demo(context, sentence):

    start_time = time.time()
    paraphaser_messages = paraphaser.build_messages(context, sentence)
    paraphaser_response = paraphaser.send_request(paraphaser_messages)  # a sentence
    query_generator_messages = query_generator.build_messages(paraphaser_response)
    query_generator_response = query_generator.send_request(query_generator_messages)
    query_list = ast.literal_eval(query_generator_response)

    with ThreadPoolExecutor() as executor:
        # 使用 map 保持顺序
        answer_list = list(executor.map(lambda query: check_fact(query, sentence), query_list))

    final_prompt = 'statement:\n' + paraphaser_response + '\n\nkey points:\n'
    for i in range(len(answer_list)):
        final_prompt += query_list[i] + '\n' + answer_list[i] + '\n\n'

    final_messages = answer_generator.build_messages(final_prompt)
    final_response = answer_generator.send_request(final_messages)
    end_time = time.time()
    print('\nfinal response:\n' + final_response)
    print('total time:', end_time - start_time)


    return final_response

def check_fact_2(query, sentence, client):
    # 构造 prompt
    prompt = "statement:\n" + sentence + "\n\nquestion:\n" + query


    fact_checker_response = fact_checker.create_chat(prompt, client)

    return fact_checker_response

def produce_input_demo_2(context, sentence, client):
    print("context:", context)
    print("sentence:", sentence)
    start_time = time.time()
    paraphaser_response = paraphaser.create_chat(context, sentence, client)  # a sentence
    query_generator_response = query_generator.create_chat(paraphaser_response,client)
    query_list = ast.literal_eval(query_generator_response)

    with ThreadPoolExecutor() as executor:
        # 使用 map 保持顺序
        answer_list = list(executor.map(lambda query: check_fact_2(query, sentence, client), query_list))

    final_prompt = 'statement:\n' + paraphaser_response + '\n\nkey points:\n'
    for i in range(len(answer_list)):
        final_prompt += query_list[i] + '\n' + answer_list[i] + '\n\n'

    final_response = answer_generator.create_chat(final_prompt, client)
    end_time = time.time()
    print('\nfinal response:\n' + final_response)
    print('total time:', end_time - start_time)


    return final_response, end_time - start_time

import nltk
nltk.download('punkt')  # 下载 punkt 数据包，包含分句所需的信息

def split_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences


import pandas as pd
if __name__ == '__main__':
    while True:
        sentence = input("sentence:\n ")
        sentences = split_paragraph(sentence)
        context = ''
        for sentence in sentences:
            produce_input_demo_2(context, sentence)
            context += sentence + ' '