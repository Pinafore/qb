import requests
import ast
from openai import OpenAI
class AnswerGenerator:
    def __init__(self):

        return

    def build_messages(self,statement):
        messages = [{"role": "system",
                     "content": "In this conversation, you will perform a fact-checking task. You will receive a statement, and we have already researched key points in the statement that are worth verifying. Please determine whether there is any information that directly contradicts the original statement. If there is, provide a brief correction starting with 'Um, actually,' that is concise and directly addresses the specific error. Avoid providing long explanations or additional details. If there is no contradiction, simply respond with 'Correct.'"}]

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

    def create_chat(self, text, client):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "In this conversation, you will perform a fact-checking task. You will receive a statement, and we have already researched key points in the statement that are worth verifying. Please determine whether there is any information that directly contradicts the original statement. If there is, provide a brief correction starting with 'Um, actually,' that is concise and directly addresses the specific error. Avoid providing long explanations or additional details. If there is no contradiction, simply respond with 'Correct.'"},

                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="gpt-4o",
            logprobs=True
        )
        log_probs = []
        for con in chat_completion.choices[0].logprobs.content:
            log_probs.append(con.logprob)
        confidence = sum(log_probs) / len(log_probs) if log_probs else -100  # Default low confidence

        return chat_completion.choices[0].message.content, confidence

if __name__ == '__main__':
    api_key = 'sk-onjTqMUfU5ZHzLdC440UYydenU74ZfRiLU985ROpw0BNQoEN'

    client = OpenAI(base_url='https://api.openai-proxy.org/v1', api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."},

            {
                "role": "user",
                "content": "Hello!",
            }
        ],
        model="gpt-4o",
        logprobs=True
    )
    print(chat_completion)
    for entry in chat_completion.choices:
        print(entry.logprobs.content)
    log_probs = []
    for con in chat_completion.choices[0].logprobs.content:
        log_probs.append(con.logprob)


    confidence = sum(log_probs) / len(log_probs) if log_probs else -100  # Default low confidence
    print(confidence)

