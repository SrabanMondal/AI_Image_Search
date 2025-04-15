import gradio as gr
import asyncio
from src.search.query_processor import QueryProcessor
from src.search.image_searcher import ImageSearcher
from src.data.image_fetcher import ImageFetcher
from src.utils.logger import setup_logger
from PIL import Image

class GradioInterface:
    def __init__(self, config, clip_model, blip_model, llm_model):
        self.config = config
        self.clip_model = clip_model
        self.blip_model = blip_model
        self.llm_model = llm_model
        self.fetcher = ImageFetcher(config["data"]["image_dir"])
        self.query_processor = QueryProcessor(llm_model, blip_model)
        self.logger = setup_logger()
        self.current_query = None
        self.current_results = []

    async def search_images(self, query):
        """Run the search pipeline for a given query."""
        self.current_query = query
        self.logger.info(f"Processing query: {query}")
        queries = self.query_processor.enhance_initial_query(query)
        self.logger.info(f"Enhanced queries: {queries}")
        image_paths = await self.fetcher.fetch_images(
            queries, self.config["data"]["max_results"]
        )
        if not image_paths:
            return [], "No images found. Try a different query."
        
        searcher = ImageSearcher(self.clip_model, image_paths, self.config["data"]["batch_size"])
        self.current_results = searcher.search(query)
        gallery = [(Image.open(path), f"Score: {score:.4f}") for path, score in self.current_results]
        return gallery, f"Found {len(gallery)} images."

    async def refine_with_feedback(self, feedback):
        """Refine query based on user feedback."""
        if not self.current_query:
            return [], "Please run a search first."
        refined_query = self.query_processor.refine_with_feedback(self.current_query, feedback)
        return await self.search_images(refined_query)

    async def refine_with_image(self, selected_image_idx):
        """Refine query based on selected image."""
        if not self.current_results or selected_image_idx is None:
            return [], "Please run a search and select an image."
        try:
            selected_image_path = self.current_results[selected_image_idx][0]
            refined_query = self.query_processor.enhance_with_image(self.current_query, selected_image_path)
            return await self.search_images(refined_query)
        except IndexError:
            return [], "Invalid image selection."

    def reset(self):
        """It clears current query and results."""
        self.current_query = None
        self.current_results = []
        return [], "Session reset. Enter a new query to start."

    def create_interface(self):
        """It defines the Gradio interface layout."""
        with gr.Blocks(title="Image Search Engine") as app:
            gr.Markdown("# Image Search Engine")
            gr.Markdown("Enter a query to search for images, refine results by selecting an image or providing feedback.")

            with gr.Row():
                query_input = gr.Textbox(label="Search Query", placeholder="e.g., 'sunset over mountains'")
                search_button = gr.Button("Search")

            gallery = gr.Gallery(label="Top Images", columns=5, height="auto")
            status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                feedback_input = gr.Textbox(label="Feedback", placeholder="e.g., 'remove trees', 'focus on blue sky'")
                feedback_button = gr.Button("Refine with Feedback")

            with gr.Row():
                image_selector = gr.Slider(minimum=0, maximum=9, step=1, label="Select Image (Index)")
                image_button = gr.Button("Refine with Image")

            reset_button = gr.Button("Reset")

            search_button.click(
                fn=self.search_images,
                inputs=query_input,
                outputs=[gallery, status]
            )
            feedback_button.click(
                fn=self.refine_with_feedback,
                inputs=feedback_input,
                outputs=[gallery, status]
            )
            image_button.click(
                fn=self.refine_with_image,
                inputs=image_selector,
                outputs=[gallery, status]
            )
            reset_button.click(
                fn=self.reset,
                outputs=[gallery, status]
            )

        return app