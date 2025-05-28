import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY must be set in .env")

client = Groq(api_key=groq_api_key)

def get_chatbot_response(question: str, context: str):
    """
    Gets a response from the Groq API based on the question and context.

    Args:
        question: The user's question.
        context: The relevant document chunks retrieved from the vector DB.

    Returns:
        The chatbot's response as a string.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": '''You are a helpful AI assistant focused on providing information based on provided documents, especially regarding business trip expenses and budget details. 
                    You can answer questions based on the document context. 
                    If a user's query seems related to budget clarification beyond the document,
                     or requires specific action related to purchases, suggest forwarding the query to the accounts team.''',
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", # You can choose a different Groq model if needed
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error interacting with Groq API: {e}")
        return "Sorry, I couldn't get a response from the language model."

def get_llm_insights(prompt: str):
    """
    Gets AI insights from the Groq API based on the analytics data.

    Args:
        prompt: The prompt containing analytics data and specific questions.

    Returns:
        The AI's insights as a string.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": '''You are an AI analytics expert focused on providing insights about trip expenses and budgets.
                    Analyze the provided data and give clear, actionable insights.
                    Focus on patterns, anomalies, and recommendations.
                    Format your response in clear sections with bullet points where appropriate.''',
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error getting LLM insights: {e}")
        return "Sorry, I couldn't generate insights from the analytics data."

# Example Usage (you can remove or comment this out later)
# if __name__ == "__main__":
#     sample_question = "What is discussed in the document?"
#     sample_context = "This document is about AI and machine learning."
#     response = get_chatbot_response(sample_question, sample_context)
#     print(f"Chatbot Response: {response}") 