from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class LocalLLM:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def enhance_query(self, user_query, num_variations=5):
        prompt = f"""
        Enhance this user query for an image search engine to find visually relevant images.
        Provide {num_variations} diverse and detailed search phrases separated by semicolons.
        User query: '{user_query}'
        Refined queries:
        """
        response = self.generate(prompt)
        queries = response.split(";")
        return [q.strip() for q in queries if q.strip()][:num_variations]

    def refine_with_feedback(self, user_query, feedback):
        prompt = f"""
        Refine this image search query based on user feedback.
        Original Query: {user_query}
        User Feedback: {feedback}
        Provide a single refined query that incorporates the feedback.
        Refined Query:
        """
        return self.generate(prompt)

    def enhance_with_caption(self, user_query, blip_caption):
        prompt = f"""
        Combine this user query and image description to create a single enhanced search query.
        Original Query: {user_query}
        Image Description: {blip_caption}
        Enhanced Search Query:
        """
        return self.generate(prompt)