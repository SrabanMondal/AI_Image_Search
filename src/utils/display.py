from PIL import Image
import matplotlib.pyplot as plt

def display_images(image_paths):
    for img_path, score in image_paths:
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Score: {score:.4f}")
        plt.axis("off")
        plt.show()