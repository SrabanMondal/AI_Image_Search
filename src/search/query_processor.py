from src.models.llm_model import LocalLLM
from src.models.blip_model import BLIPModel

class QueryProcessor:
    def __init__(self, llm_model, blip_model):
        self.llm_model = llm_model
        self.blip_model = blip_model

    def enhance_initial_query(self, user_query, num_variations=5):
        return self.llm_model.enhance_query(user_query, num_variations)

    def refine_with_feedback(self, user_query, feedback):
        return self.llm_model.refine_with_feedback(user_query, feedback)

    def enhance_with_image(self, user_query, image_path):
        caption = self.blip_model.generate_caption(image_path)
        if user_query:
            return self.llm_model.enhance_with_caption(user_query, caption)
        else:
            # If no user query, use caption as the query
            return caption