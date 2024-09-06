import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from app.main import identify
from app.shared.utils import upload_file

if TYPE_CHECKING:
    from fastapi import UploadFile


class TestManager:
    file_path = Path("tests/services/identify/solution_identify.json")
    with file_path.open() as file:
        expected_results: dict = json.load(file)

    @pytest.mark.asyncio
    async def test_identify_cap(self):
        folder = "tests/services/identify/images"
        imgs: list[str] = os.listdir("tests/services/identify/images")
        total_accuracy = 0
        for img in imgs:
            img_path = Path(str(folder)) / img
            file: UploadFile = await upload_file(img_path)
            identify_result: list[dict] = await identify(file=file)
            identify_result: list[str] = [cap["id"] for cap in identify_result]

            expected_result = self.expected_results[img]
            if expected_result in identify_result:
                logger.info(f"{expected_result} correctly found in {identify_result}")
                total_accuracy += 1
            else:
                logger.error(f"{expected_result} not found in {identify_result}")
        logger.info(
            f"Currently the bottle cap has an accuracy of {total_accuracy / len(imgs)} "
            f"({total_accuracy}/{len(imgs)})"
        )

        # Total DB: Almost 300
        # My model
        # 255 vector size
        # Accuracy 100: 0.86
        # Accuracy 50: 0.73
        # Accuracy 25: 0.46
        # Accuracy 10: 0.33

        # Total DB: Almost 300
        # My model
        # 2048 vector size
        # Accuracy 100: 0.66
        # Accuracy 50: 0.4
        # Accuracy 25: 0.4
        # Accuracy 10: 0.13

        # Total DB: Almost 300
        # 2048 vector size
        # Pretrained model
        # Accuracy 100: 1
        # Accuracy 50: 1
        # Accuracy 25: 1
        # Accuracy 10: 0.86

        # Total DB: Almost 300
        # 255 vector size
        # Pretrained model
        # Accuracy 100: 0.26
        # Accuracy 50: 0.26
        # Accuracy 25: 0.13
        # Accuracy 10: 0