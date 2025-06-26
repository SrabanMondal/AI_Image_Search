# Image Search Engine

![Image Search
Engine](https://img.shields.io/badge/Status-Production%20Ready-green.svg)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org)

A **production-grade image search engine** powered by state-of-the-art
AI models (CLIP, BLIP, and a local LLM) for text- and image-based
queries. This application enables users to search for images using
natural language, uploaded images, or both, with iterative refinement
via feedback or image selection. Built with a modular architecture, it
supports multi-user sessions, robust session management, and automatic
cleanup of temporary files. Deployed on Hugging Face Spaces, it delivers
a scalable, user-friendly Gradio interface with valid JPEG downloads
from a responsive gallery.

Designed with industry best practices, the project showcases expertise
in **AI/ML integration**, **asynchronous programming**, **web app
development**, and **cloud deployment**. Ideal for applications in
content discovery, e-commerce, or creative industries, this engine
demonstrates end-to-end development from model integration to production
deployment.

## 🚀 Features

-   **Text-Based Search**: Query images using natural language (e.g.,
    "sunset over mountains") with CLIP-based semantic ranking.
-   **Image-Based Search**: Upload images to generate BLIP-driven
    captions, combined with optional text queries for precise results.
-   **Iterative Refinement**: Refine searches via user feedback (e.g.,
    "focus on blue sky") or by selecting gallery images, leveraging LLM
    query enhancement.
-   **Multi-User Support**: Isolated sessions with unique IDs, ensuring
    concurrent users have independent queries and temporary files.
-   **Responsive Gallery**: Displays up to 10 ranked images with scores,
    with downloadable, valid JPEGs via Gradio's gallery component.
-   **Session Management**: Persists query state across interactions,
    with automatic cleanup of temporary files after 30 minutes of
    inactivity.
-   **Production-Ready Deployment**: Hosted on Hugging Face Spaces with
    a live URL, optimized for scalability and reliability.
-   **Modular Architecture**: Clean, maintainable code with separated
    concerns (models, data, search, UI, utilities).

## 🛠️ Tech Stack

-   **Backend**: Python 3.10, asyncio for asynchronous image fetching
    and processing.
-   **AI/ML Models**:
    -   **CLIP (openai/clip-vit-base-patch32)**: Semantic image-text
        matching for ranking.
    -   **BLIP (microsoft/git-large)**: Image captioning for query
        enhancement.
    -   **Local LLM (distilgpt2)**: Query refinement and natural
        language processing.
-   **Frontend**: Gradio 3.50.0 for an interactive, web-based UI.
-   **Data Fetching**: DuckDuckGo API for image retrieval, with local
    caching in `data/images/`.
-   **Dependencies**: PyTorch, Transformers, Pillow, aiohttp, PyYAML,
    and more (see `requirements.txt`).
-   **Deployment**: Hugging Face Spaces (Linux, Python 3.10) with
    Git-based CI/CD.
-   **Utilities**: Custom logging, session management, and configuration
    via YAML.

## 📋 Prerequisites

-   **Python**: 3.10
-   **Git**: For cloning and managing the repository.
-   **Hugging Face Account**: For deployment (optional for local use).
-   **Hardware**: CPU (GPU optional for faster inference).
-   **Disk Space**: \~2GB for models and cached images.

## 🏗️ Installation

1.  **Clone the Repository**:

    ``` bash
    git clone https://github.com/SrabanMondal/AI_Image_Search.git
    cd image-search-engine
    ```

2.  **Set Up a Virtual Environment** (recommended):

    ``` bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:

    ``` bash
    pip install -r requirements.txt
    ```

4.  **Configure Settings**:

    -   Edit `config/app_config.yaml` to adjust:
        -   Model paths (e.g., `clip: openai/clip-vit-base-patch32`).
        -   Data directories (`data/images/`, `data/temp/`).
        -   Session timeout (default: 1800 seconds).
        -   Max results (default: 20 images).

5.  **Run Locally**:

    ``` bash
    python app.py
    ```

    -   Access the app at `http://0.0.0.0:7860`.

## 🚀 Usage

1.  **Open the Interface**:
    -   Local: `http://0.0.0.0:7860`
    -   Live:
        `https://huggingface.co/spaces/srabanmondal/Image-Search-Engine`
        (after deployment).
2.  **Search for Images**:
    -   **Text Search**: Enter a query (e.g., "sunset over mountains")
        and click "Search with Text".
    -   **Image Search**: Upload an image (max 5 per session) and
        optionally add a text query, then click "Search with
        Image/Text".
    -   View up to 10 ranked images in the gallery with similarity
        scores.
3.  **Refine Results**:
    -   **Feedback**: Provide natural language feedback (e.g., "remove
        trees") and click "Refine with Feedback".
    -   **Image Selection**: Choose an image index (0-9) and click
        "Refine with Image" to enhance the query using BLIP captions.
4.  **Download Images**:
    -   Hover over gallery images and click the download button to save
        valid JPEG files.
5.  **Reset Session**:
    -   Click "Reset" to clear queries, results, and temporary files.
6.  **Multi-User Support**:
    -   Concurrent users get isolated sessions with unique IDs.
    -   Temporary files are stored in `data/temp/` and cleaned up after
        30 minutes of inactivity.

## 🏛️ Architecture

The project follows a modular, layered architecture for maintainability
and scalability:

-   **Models (`src/models/`)**:
    -   `CLIPModel`: Embeds images and text for similarity ranking.
    -   `BLIPModel`: Generates captions from uploaded images.
    -   `LocalLLM`: Enhances and refines queries using distilgpt2.
-   **Data (`src/data/`)**:
    -   `ImageFetcher`: Asynchronously fetches images via DuckDuckGo
        API, caching in `data/images/`.
    -   `data_utils`: Handles image processing and storage.
-   **Search (`src/search/`)**:
    -   `QueryProcessor`: Enhances queries using LLM and BLIP.
    -   `ImageSearcher`: Ranks images using CLIP embeddings.
-   **Interfaces (`src/interfaces/`)**:
    -   `GradioInterface`: Builds the web UI with Gradio Blocks,
        supporting text/image inputs, gallery output, and refinement.
-   **Utilities (`src/utils/`)**:
    -   `SessionManager`: Manages user sessions, query state, and temp
        file cleanup.
    -   `config`: Loads settings from `app_config.yaml`.
    -   `logger`: Custom logging for debugging and monitoring.
-   **Entry Point (`app.py`)**:
    -   Initializes models and Gradio interface, designed for Hugging
        Face Spaces deployment.


## 📈 Performance Optimization

-   **Asynchronous Fetching**: Uses `aiohttp` for concurrent image
    downloads, reducing latency.
-   **Batch Processing**: CLIP processes images in batches (configurable
    in `app_config.yaml`).
-   **Session Cleanup**: Automatic deletion of temporary files after 30
    minutes (configurable).
-   **Scalability**: Gradio's queue system handles concurrent users,
    with upload limits (5 images/session) to prevent abuse.

For high traffic: - Upgrade to CPU Upgraded or GPU on Hugging Face
Spaces. - Reduce `max_results` in `app_config.yaml` (e.g., from 20 to
10). - Cache model embeddings for faster inference (future work).


## 🙌 Acknowledgments

-   **Hugging Face**: For Transformers, CLIP, and Spaces hosting.
-   **OpenAI**: For the CLIP model implementation.
-   **Salesforce**: For the BLIP model.
-   **DuckDuckGo**: For the free image search API.


------------------------------------------------------------------------

*Built with 💡 and ☕ to empower visual discovery through AI.*
