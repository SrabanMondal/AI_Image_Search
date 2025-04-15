import google.generativeai as genai
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

SECRET_KEY = os.getenv('GEMINI_API')
class LocalLLM:
    def __init__(self,model_name, device="cpu"):
        api_key = SECRET_KEY
        model_name="gemini-2.0-flash"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt, max_length=1000):
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def enhance_query(self, user_query, num_variations=5):
        prompt = f"""
        Enhance this user query for an image search engine to find visually relevant images.
        Provide {num_variations} diverse and detailed search phrases.
        Guidelines
        - Give clean text
        - Don't add extra text
        - Give search queries separated by ;
        User query: '{user_query}'
        Refined queries:
        """
        response = self.generate(prompt)
        queries = response.split(";")
        return [q.strip() for q in queries if q.strip()][:num_variations]

    def refine_with_feedback(self, user_query, feedback):
        prompt = f"""
        You are an AI that refines image search queries based on user feedback. 
        Given the initial query, you will enhance the query again 
        while incorporating the user's feedback.

        **Original Query:** {user_query}   
        **User Feedback:** {feedback}  

        Based on this feedback, refine the query to remove unwanted details, highlight preferred details, 
        and make the search more accurate. Ensure the new query remains detailed and descriptive.
        Focus on giving search query that can help image search engine like duckduckgo fetch more relevant images.
        **Refined Query:**  
        """
        return self.generate(prompt)

    def enhance_with_caption(self, user_query, blip_caption):
        prompt = f"""
        You are an AI specializing in refining search queries by intelligently combining a user's input query and a detailed image description.
        Your goal is to generate multiple highly relevant, varied search queries for an image search engine that can help search engine to fetch most relevant images.

        ### Input:
        - **Original Query:** "{user_query}"
        - **Expanded Image Description:** "{blip_caption}"

        ### Guidelines:
        - Maintain the core intent of the original query.
        - Incorporate detailed elements from the description
        - Give only one search query with clean text
        - Focus on giving good search query that can help image search engine like duckduckgo fetch relevant images
        ### Enhanced Search Query:
        """
        return self.generate(prompt)
