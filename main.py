from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.clip_model import CLIPModel
from src.models.blip_model import BLIPModel
from src.models.llm_model import LocalLLM
from src.interfaces.gradio_interface import GradioInterface

def main():
    config = load_config()
    logger = setup_logger()
    logger.info("Initializing models...")
    clip_model = CLIPModel(config["models"]["clip"], device="cpu")
    blip_model = BLIPModel(config["models"]["blip"], device="cpu")
    llm_model = LocalLLM(config["models"]["llm"], device="cpu")
    interface = GradioInterface(config, clip_model, blip_model, llm_model)
    logger.info("Launching Gradio interface...")
    app = interface.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()