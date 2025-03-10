import requests
class paraphaser:
    def __init__(self):

        return

    def build_messages(self,context,target_sentence):
        messages = [{"role": "system",
                     "content": "In this task, you are given two inputs: a **context** and a **target sentence**. Your goal is to transform the target sentence into a standalone, fact-checkable sentence. To achieve this:\n"
                                "1. Combine the **context** and the **target sentence** where necessary, ensuring the target sentence makes sense without relying on the context. \n "
                                "2. Replace pronouns (e.g., 'it,' 'he,' 'she') in the target sentence with the specific nouns they refer to, using information from the context. \n "
                                "3. Do not omit any details from the target sentence. Retain all the information while ensuring clarity. \n "
                                "4. If context is empty, return the target sentence unchanged.\n"
                                "Input format:  \n"
                                "Context: [context text]  \n"
                                "Target Sentence: [sentence to be processed]  \n"
                                "Output format:  \n"
                                "[Transformed standalone sentence]  "
                                "Examples:  \n"
                                "Input 1:  \n"
                                "Context: In Harry Potter, he is a wizard and the main character of the series.  \n"
                                "Target Sentence:** He is known as the Boy Who Lived.  \n"
                                "Output 1:  \n"
                                "In Harry Potter, Harry Potter is known as the Boy Who Lived. \n "
                                "Input 2:  "
                                "Context: In Harry Potter, Hermione Granger is one of Harry's closest friends and is highly intelligent. \n"
                                "Target Sentence: She is often referred to as the brightest witch of her age. \n "
                                "Output 2:  "
                                "In Harry Potter, Hermione Granger is often referred to as the brightest witch of her age. \n "
                                "Input 3:  "
                                "Context:\n  "
                                "Target Sentence:  The Amazon rainforest is the largest tropical rainforest in the world. \n "
                                "Output 3:  \n"
                                "The Amazon rainforest is the largest tropical rainforest in the world.  \n"
                                "Input 4: \n "
                                "Context: The movie “Predator,” aside from being generally cool, is notable in other ways. \n "
                                "Target Sentence: It was #1 at the box office for its opening weekend, won an Academy Award for Visual Effects and has the notable privilege of being the only movie to feature two future Governors: Arnold Schwarzenegger and Jesse Ventura.  \n"
                                "Output 4: \n "
                                "The movie “Predator,” aside from being generally cool, was #1 at the box office for its opening weekend, won an Academy Award for Visual Effects, and has the notable privilege of being the only movie to feature two future Governors: Arnold Schwarzenegger and Jesse Ventura."

}]
        messages.append({"role": "user", "content": "Context: "+context+"\nTarget Snetence: "+target_sentence})
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

    def create_chat(self, context, target_sentence, client):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "In this task, you are given two inputs: a **context** and a **target sentence**. Your goal is to transform the target sentence into a standalone, fact-checkable sentence. To achieve this:\n"
                                "1. Combine the **context** and the **target sentence** where necessary, ensuring the target sentence makes sense without relying on the context. \n "
                                "2. Replace pronouns (e.g., 'it,' 'he,' 'she') in the target sentence with the specific nouns they refer to, using information from the context. \n "
                                "3. Do not omit any details from the target sentence. Retain all the information while ensuring clarity. \n "
                                "4. If context is empty, return the target sentence unchanged.\n"
                                "Input format:  \n"
                                "Context: [context text]  \n"
                                "Target Sentence: [sentence to be processed]  \n"
                                "Output format:  \n"
                                "[Transformed standalone sentence]  "
                                "Examples:  \n"
                                "Input 1:  \n"
                                "Context: In Harry Potter, he is a wizard and the main character of the series.  \n"
                                "Target Sentence:** He is known as the Boy Who Lived.  \n"
                                "Output 1:  \n"
                                "In Harry Potter, Harry Potter is known as the Boy Who Lived. \n "
                                "Input 2:  "
                                "Context: In Harry Potter, Hermione Granger is one of Harry's closest friends and is highly intelligent. \n"
                                "Target Sentence: She is often referred to as the brightest witch of her age. \n "
                                "Output 2:  "
                                "In Harry Potter, Hermione Granger is often referred to as the brightest witch of her age. \n "
                                "Input 3:  "
                                "Context:\n  "
                                "Target Sentence:  The Amazon rainforest is the largest tropical rainforest in the world. \n "
                                "Output 3:  \n"
                                "The Amazon rainforest is the largest tropical rainforest in the world.  \n"
                                "Input 4: \n "
                                "Context: The movie “Predator,” aside from being generally cool, is notable in other ways. \n "
                                "Target Sentence: It was #1 at the box office for its opening weekend, won an Academy Award for Visual Effects and has the notable privilege of being the only movie to feature two future Governors: Arnold Schwarzenegger and Jesse Ventura.  \n"
                                "Output 4: \n "
                                "The movie “Predator,” aside from being generally cool, was #1 at the box office for its opening weekend, won an Academy Award for Visual Effects, and has the notable privilege of being the only movie to feature two future Governors: Arnold Schwarzenegger and Jesse Ventura."

},
                {
                    "role": "user",
                    "content": "Context: "+context+"\nTarget Snetence: "+target_sentence,
                }
            ],
            model="gpt-4o",
        )
        # print(chat_completion)
        return chat_completion.choices[0].message.content

from jsonl_produce import read_jsonl, write_jsonl
if __name__ == '__main__':
    Paraphaser = paraphaser()
    data = read_jsonl("../database_process/narrativeqa.jsonl")
    id = int(input("enter id:"))
    iid = 1
    import time

    max_retries = 30  # 最大重试次数
    retry_delay = 2  # 重试延迟时间（秒）

    for line in data:
        if iid < id:
            iid += 1
            continue

        retries = 0
        while retries < max_retries:
            try:
                messages = Paraphaser.build_messages(line['context'], line['sentence'])
                response = Paraphaser.send_request(messages)
                print("context: " + line['context'] + "\nsentence: " + line['sentence'] + "\nresponse: " + response)
                line['response'] = response
                write_jsonl("../database_process/atomic_sentences.jsonl", [line])
                break  # 如果成功，跳出重试循环
            except Exception as e:
                print(f"Error occurred: {e}. Retrying ({retries + 1}/{max_retries})...")
                retries += 1
                time.sleep(retry_delay)  # 等待一段时间后重试
        else:
            print(f"Failed to process line after {max_retries} retries.")
