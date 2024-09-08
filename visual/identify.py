import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.main import identify
from app.shared.utils import upload_file

if TYPE_CHECKING:
    from fastapi import UploadFile


async def create_composite(img_path: Path, result: list[tuple], output_path: Path) -> None:  # noqa: D103
    main_image = Image.open(img_path)
    main_image = main_image.resize((300, 300))

    num_images = len(result)
    grid_size = int(np.ceil(num_images**0.5))

    additional_image_size = 100
    # Composite canvas adjusts width and height dynamically based on grid
    composite_width = main_image.width + additional_image_size * grid_size + 50
    composite_height = max(main_image.height, additional_image_size * (grid_size)) + 50
    composite_image = Image.new("RGB", (composite_width, composite_height), (255, 255, 255))

    # Centering the main image
    main_image_x = (
        composite_width - main_image.width
    ) // 2 - additional_image_size * grid_size // 2
    composite_image.paste(main_image, (main_image_x, (composite_height - main_image.height) // 2))

    # Adjust offsets to center the additional images
    x_offset = main_image_x + main_image.width + 10
    y_offset = 10

    draw = ImageDraw.Draw(composite_image)
    font = ImageFont.load_default()

    for i, (img, score) in enumerate(result):
        x = x_offset + (i % grid_size) * (additional_image_size + 10)
        y = y_offset + (i // grid_size) * (additional_image_size + 10)
        composite_image.paste(img, (x, y))

        score_text = f"{score:.2f}"
        bbox = draw.textbbox((x, y), score_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (additional_image_size - text_width) // 2
        text_y = y + additional_image_size - text_height

        background_padding = 6
        background_x0 = text_x - background_padding
        background_y0 = text_y - background_padding
        background_x1 = text_x + text_width + background_padding
        background_y1 = text_y + text_height + background_padding

        draw.rectangle([background_x0, background_y0, background_x1, background_y1], fill="black")
        draw.text((text_x, text_y), score_text, fill="yellow", font=font)

    composite_image.save(output_path)
    composite_image.show()


async def main():  # noqa: D103
    img_path: Path = Path("tests") / "services" / "identify" / "images" / "11.jpg"
    file: UploadFile = await upload_file(img_path)
    res = await identify(file)

    folder_path = Path("database") / "caps"
    additional_images = [
        (Image.open(folder_path / path["id"]).resize((100, 100)), path["score"]) for path in res
    ]
    output_path = Path("visual") / "result" / "identify.jpg"
    await create_composite(img_path, additional_images, output_path)


if __name__ == "__main__":
    asyncio.run(main())
