import requests
import ast
class FactChecker:
    def __init__(self):

        return

    def build_messages(self,statement):
        messages = [{"role": "system",
                     "content": "In this conversation, you will perform a fact-checking task. You will receive a statement and a question based on that statement. Please answer the question and give a brief explanation, without saying anything else, and ensure that your answer is as factually accurate as possible."}]

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
                    "content": "In this conversation, you will perform a fact-checking task. You will receive a statement and a question based on that statement. Please answer the question and give a brief explanation, without saying anything else, and ensure that your answer is as factually accurate as possible."},

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
    Factchecker =   FactChecker()
    statement = "Unown, the symbol Pokémon, is a psychic-type Pokémon first discovered in the Ruins of Alph, that is only capable of learning one move: Hidden Power. Unown's primary appeal is as an alphabet. Though it has no evolutions, this one Pokémon has 26 different forms, one for each letter of the Roman alphabet."
    messages=Factchecker.build_messages(statement)
    response = Factchecker.send_request(messages)
    print(response)
    lis = ast.literal_eval(response)
    print(lis)
