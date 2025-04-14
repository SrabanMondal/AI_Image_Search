from src.search.query_processor import QueryProcessor
from src.search.image_searcher import ImageSearcher
from src.data.image_fetcher import ImageFetcher
from src.utils.display import display_images
from src.utils.logger import setup_logger
import asyncio

class CLIInterface:
    def __init__(self, config, clip_model, blip_model, llm_model):
        self.config = config
        self.fetcher = ImageFetcher(config["data"]["image_dir"])
        self.query_processor = QueryProcessor(llm_model, blip_model)
        self.logger = setup_logger()

    async def run(self):
        query = input("Enter your image search query: ")
        while True:
            queries = self.query_processor.enhance_initial_query(query)
            self.logger.info(f"Enhanced queries: {queries}")
            image_paths = await self.fetcher.fetch_images(
                queries, self.config["data"]["max_results"]
            )
            if not image_paths:
                print("No images found. Try a different query.")
                query = input("Enter a new query or 'exit' to quit: ")
                if query.lower() == "exit":
                    break
                continue

            searcher = ImageSearcher(None, image_paths, self.config["data"]["batch_size"])
            results = searcher.search(query)
            print("\nTop images:")
            display_images(results)

            action = input(
                "Choose an action: (1) Select image, (2) Provide feedback, (3) New query, (4) Exit: "
            )
            if action == "1":
                idx = int(input("Enter the index of the selected image (0-9): "))
                selected_image = results[idx][0]
                query = self.query_processor.enhance_with_image(query, selected_image)
            elif action == "2":
                feedback = input("Enter feedback (e.g., 'remove X', 'focus on Y'): ")
                query = self.query_processor.refine_with_feedback(query, feedback)
            elif action == "3":
                query = input("Enter a new query: ")
            elif action == "4":
                break