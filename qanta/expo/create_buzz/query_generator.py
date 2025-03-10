import requests
import ast
from openai import OpenAI
class query_generator:
    def __init__(self):

        return

    def build_messages(self,statement):
        messages = [{"role": "system",
                     "content": "In this conversation, you will receive a statement. You need to identify each fact within the statement that requires verification and generate a separate question for each. Return the list of questions.\n"
                                "Answer format:\n"
                                "[ \"<question1>\" , \"<question2> \", ... ]\n"
                                "Please return a list(in grammar of python) containing questions, as shown in the example and don't say anything else. Please use single quotes for each quotation mark in the question, and enclose each question in double quotes."}]

        messages.append({"role": "user", "content": statement})
        return messages


    def send_request(self,messages):
        url = "https://api.f2gpt.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-f2czNY6Be0Yk507RjQNPWJrAMTNXHmHJiqpxSfk231pvYGdA"
        }
        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        return response_json['choices'][0]['message']['content']

    def create_chat(self, text, client=None):

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": '''In this conversation, you will receive a statement. You need to identify each fact within the statement that requires verification and generate a separate question for each. Return these questions. Each line in your answer should contain a question. Don't say anything else
                    
                                Answer format:
                                <question 1> 
                                <question 2>
                                ...
                                <question n> 
                                '''},

                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="gpt-4o",
        )
        # print(chat_completion)
        return chat_completion.choices[0].message.content

if __name__ == '__main__':
    Query_generator = query_generator()
    statement = '*Isn\'t it enough to know that I ruined a pony, making a gift for you?* Oh hi everybody, it\'s me, Jonathan Coulton. That was a snippet from my song "Skullcrusher Mountain," in which a mad scientist creates a monster by mixing monkeys and ponies together. You know, when you think about it, a lot of monsters are just two creatures mixed together, like griffins are eagles and lions, hippocampi are eagles and fish, and centaurs are horses and people.'
    messages=Query_generator.build_messages(statement)
    response = Query_generator.send_request(messages)
    print(response)
    lis = ast.literal_eval(response)
    print(lis)
