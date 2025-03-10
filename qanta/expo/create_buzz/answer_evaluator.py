import requests
import ast
from openai import OpenAI
class Evaluator:
    def __init__(self):

        return

    def build_messages(self, sentence1, sentence2, statement):
        messages = [{"role": "system",
                     "content": "In this conversation, you will receive a statement along with two corrections to it. Please determine whether the two corrections convey a relatively similar meaning. If they do, answer 'Y'. If they do not, answer 'N'. Do not provide any explanation."}]

        messages.append({"role": "user", "content": f"Statement:{statement}\nCorrection 1: {sentence1}\nCorrection 2: {sentence2}"})
        return messages


    def send_request(self,messages):
        url = "https://api.f2gpt.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-f2YvUdLjaOLgQwcvt4hwHXEdzpyJHx2T70le4noX08dYplkW"
        }
        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        return response_json['choices'][0]['message']['content']

    def create_chat(self, sentence1, sentence2, statement, client):


        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "In this conversation, you will receive a statement along with two corrections to it. Please determine whether the two corrections convey a relatively similar meaning. If they do, answer 'Y'. If they do not, answer 'N'. Do not provide any explanation."},

                {
                    "role": "user",
                    "content": f"Statement:{statement}\nCorrection 1: {sentence1}\nCorrection 2: {sentence2}",
                }
            ],
            model="gpt-4o",
        )

        # print(chat_completion)
        return chat_completion.choices[0].message.content