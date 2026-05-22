
from ollama import chat

response = chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'can you give all the stents OPCS-4 Codes?'}
    ]
)

print(response['message']['content'])