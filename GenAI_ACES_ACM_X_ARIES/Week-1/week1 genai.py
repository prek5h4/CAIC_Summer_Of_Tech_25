import os
from groq import Groq
import google.generativeai as genai
client = Groq(
    api_key= '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',
)
genai.configure(api_key = 'AIzaSyDDvh4ciTziBe_8uwm4HxM3zMiQUWPlkYE')

gmodel = genai.GenerativeModel('gemini-1.5-flash')


def run_groq_model(prompt, model_name="llama3-8b-8192"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def run_gemini_model(prompt):
    response = gmodel.generate_content(prompt)
    return response.text

def run_chat_agent():
    print("Bot: Hello User, How can I help you today?")
    while True:
        input_text = input("User: ")
        if input_text.lower() in ['exit', 'quit']:
            print("Bot: Thank you for your time")
            break

        if "correct" in input_text.lower():
            response = run_gemini_model(input_text)
            print("Gemini Bot:", response)
        else:
            response = run_groq_model(input_text)
            print("Groq Bot:",response)
        
   


run_chat_agent()

        