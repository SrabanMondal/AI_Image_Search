import os
import aiohttp
import asyncio
from duckduckgo_search import DDGS
from src.utils.logger import setup_logger

class ImageFetcher:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        self.logger = setup_logger()

    async def fetch_image_urls(self, query, max_results=20):
        urls = []
        try:
            with DDGS() as ddgs:
                results = ddgs.images(query, max_results=max_results)
                urls = [img["image"] for img in results if img["image"].endswith(".jpg")]
        except Exception as e:
            self.logger.error(f"Error fetching images for {query}: {e}")
        return urls

    async def download_image(self, session, url, save_path):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    image_data = await response.read()
                    with open(save_path, "wb") as f:
                        f.write(image_data)
                    return save_path
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
        return None

    async def fetch_images(self, queries, max_results=20):
        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_image_urls(query, max_results) for query in queries]
            all_urls = await asyncio.gather(*tasks)
            image_urls = [url for urls in all_urls for url in urls]
            tasks = [
                self.download_image(session, url, f"{self.image_dir}/img_{i}.jpg")
                for i, url in enumerate(image_urls)
            ]
            results = await asyncio.gather(*tasks)
            return [path for path in results if path]