import asyncio
from io import BytesIO
from pathlib import Path

import aiofiles
import matplotlib.pyplot as plt
import numpy as np
from fastapi import UploadFile
from loguru import logger
from matplotlib import animation
from PIL import Image, ImageDraw

from app.main import detect


async def _detect_animation(file_path: Path, output_path: Path) -> None:
    async with aiofiles.open(file_path, mode="rb") as file:
        file_contents = await file.read()
        upload_file = UploadFile(filename=str(file_path), file=BytesIO(file_contents))

    rectangles: list[tuple] = await detect(upload_file)

    image = Image.open(file_path)
    image_np = np.array(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")

    def update(frame):
        img_copy = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img_copy)
        for i in range(frame + 1):
            (x, y, w, h) = rectangles[i]
            center_x = x + w / 2
            center_y = y + h / 2
            radius = min(w, h) / 2
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                         outline="yellow", width=10)

        img_np_frame = np.array(img_copy)
        ax.imshow(img_np_frame)
        return ax

    ani = animation.FuncAnimation(fig, update, frames=len(rectangles), repeat=False)
    ani.save(output_path, writer="pillow")
    logger.info(f"Saved animation to {output_path}")


async def process_directory(directory: Path, output_directory: Path) -> None:
    """Create the animation for the detect method.

    Args:
    ----
        directory: Where to get the images to analyze.
        output_directory: Where to save.

    """
    for filename in directory.iterdir():
        output_path = output_directory / (filename.stem + "_animation.gif")
        await _detect_animation(filename, output_path)


if __name__ == "__main__":
    input_directory = Path("database/test-images/test-i-have")
    output_directory = Path("animations")
    output_directory.mkdir(parents=True, exist_ok=True)
    asyncio.run(process_directory(input_directory, output_directory))
