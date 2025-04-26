import gradio as gr
import asyncio
import os
import time
from src.search.query_processor import QueryProcessor
from src.search.image_searcher import ImageSearcher
from src.data.image_fetcher import ImageFetcher
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager
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
        self.temp_dir = config["data"]["temp_dir"]
        self.session_manager = SessionManager(self.temp_dir, config["session"]["timeout_seconds"])
        os.makedirs(self.temp_dir, exist_ok=True)

    async def search_images(self, query, session_id):
        """Run the search pipeline for a given query."""
        self.session_manager.update_session_activity(session_id)
        self.logger.info(f"Processing query: {query} for session {session_id}")
        queries = self.query_processor.enhance_initial_query(query)
        self.logger.info(f"Enhanced queries: {queries}")
        image_paths = await self.fetcher.fetch_images(
            queries, self.config["data"]["max_results"]
        )
        if not image_paths:
            self.session_manager.update_session_data(session_id, current_query=query, current_results=[])
            return [], "No images found. Try a different query.", session_id
        
        searcher = ImageSearcher(self.clip_model, image_paths, self.config["data"]["batch_size"])
        results = searcher.search(query)
        self.session_manager.update_session_data(session_id, current_query=query, current_results=results)
        gallery = [(path, f"Score: {score:.4f}") for path, score in results]
        return gallery, f"Found {len(gallery)} images.", session_id

    async def search_with_image(self, query, uploaded_image, session_id):
        """Run the search pipeline with an uploaded image and optional text query."""
        self.session_manager.update_session_activity(session_id)
        if uploaded_image is None:
            return [], "Please upload an image.", session_id
        
        # Save uploaded image with unique filename
        temp_image_name = f"{session_id}_{int(time.time())}.jpg"
        temp_image_path = os.path.join(self.temp_dir, temp_image_name)
        uploaded_image.save(temp_image_path)
        self.session_manager.add_temp_file(session_id, temp_image_path)
        
        # Enhance query with image
        current_query, _ = self.session_manager.get_session_data(session_id)
        enhanced_query = self.query_processor.enhance_with_image(query or current_query or "", temp_image_path)
        self.logger.info(f"Enhanced query from image: {enhanced_query} for session {session_id}")
        
        # Run search with enhanced query
        gallery, status, session_id = await self.search_images(enhanced_query, session_id)
        return gallery, status, session_id

    async def refine_with_feedback(self, feedback, session_id):
        """Refine query based on user feedback."""
        self.session_manager.update_session_activity(session_id)
        if not feedback:
            return [], "Please provide feedback.", session_id
        current_query, _ = self.session_manager.get_session_data(session_id)
        if not current_query:
            return [], "Please run a search first.", session_id
        refined_query = self.query_processor.refine_with_feedback(current_query, feedback)
        gallery, status, session_id = await self.search_images(refined_query, session_id)
        return gallery, status, session_id

    async def refine_with_image(self, selected_image_idx, session_id):
        """Refine query based on selected image."""
        self.session_manager.update_session_activity(session_id)
        current_query, current_results = self.session_manager.get_session_data(session_id)
        if not current_results:
            return [], "Please run a search first.", session_id
        try:
            selected_image_path = current_results[selected_image_idx][0]
            refined_query = self.query_processor.enhance_with_image(current_query or "", selected_image_path)
            gallery, status, session_id = await self.search_images(refined_query, session_id)
            return gallery, status, session_id
        except IndexError:
            return [], "Invalid image selection.", session_id

    def reset(self, session_id):
        """Clear session data and temporary files."""
        self.session_manager.cleanup_session(session_id)
        # Recreate session to allow immediate reuse
        new_session_id = self.session_manager.create_session()
        return [], "Session reset. Enter a new query or upload an image to start.", new_session_id

    def create_session(self):
        """Create a new session for a user."""
        return self.session_manager.create_session()

    def create_interface(self):
        """Define the Gradio interface layout."""
        with gr.Blocks(title="Image Search Engine") as app:
            gr.Markdown("# Image Search Engine")
            gr.Markdown("Enter a query and/or upload an image to search for similar images. Refine results by selecting an image or providing feedback.")
            
            session_id = gr.State(value=None)  # Store session ID per user
            
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Search Query (Optional)", placeholder="e.g., 'sunset over mountains'")
                    image_input = gr.Image(type="pil", label="Upload Image (Optional)")
                with gr.Column():
                    search_button = gr.Button("Search with Text")
                    image_search_button = gr.Button("Search with Image/Text")

            gallery = gr.Gallery(label="Top Images", columns=5, height="auto")
            status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                feedback_input = gr.Textbox(label="Feedback", placeholder="e.g., 'remove trees', 'focus on blue sky'")
                feedback_button = gr.Button("Refine with Feedback")

            with gr.Row():
                image_selector = gr.Slider(minimum=0, maximum=9, step=1, label="Select Image (Index)")
                image_button = gr.Button("Refine with Image")

            reset_button = gr.Button("Reset")

            # Initialize session on page load
            app.load(
                fn=self.create_session,
                inputs=None,
                outputs=session_id
            )

            # Bind actions to buttons
            search_button.click(
                fn=self.search_images,
                inputs=[query_input, session_id],
                outputs=[gallery, status, session_id]
            )
            image_search_button.click(
                fn=self.search_with_image,
                inputs=[query_input, image_input, session_id],
                outputs=[gallery, status, session_id]
            )
            feedback_button.click(
                fn=self.refine_with_feedback,
                inputs=[feedback_input, session_id],
                outputs=[gallery, status, session_id]
            )
            image_button.click(
                fn=self.refine_with_image,
                inputs=[image_selector, session_id],
                outputs=[gallery, status, session_id]
            )
            reset_button.click(
                fn=self.reset,
                inputs=session_id,
                outputs=[gallery, status, session_id]
            )

        return app