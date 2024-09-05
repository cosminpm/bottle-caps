import csv
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from oml import datasets as d
from oml.losses import TripletLossWithMiner
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.samplers import BalanceSampler
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

from app.services.identify.pinecone_container import PineconeContainer

load_dotenv()
pinecone_container = PineconeContainer()


def create_csv():
    base_dir = Path("database") / "training"
    output_csv = "training_data.csv"

    with open(output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["path", "label"])

        counter = 0
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # label = os.path.basename(root)
                label = counter
                image_path = os.path.join(root, file)

                csvwriter.writerow([image_path, label])
            counter += 1
    logger.info(f"CSV file '{output_csv}' generated successfully.")


def training():
    train_csv_path: str = str(Path("scripts") / "training_data.csv")
    df_train = pd.read_csv(train_csv_path)

    model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
    transform, _ = get_transforms_for_pretrained("vits16_dino")

    train = d.ImageLabeledDataset(df_train, transform=transform)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
    sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)

    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logger.info(criterion.last_logs)

    model_save_path = Path("app") / "models" / "trained_model.pth"
    #torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")




if __name__ == "__main__":
    training()
