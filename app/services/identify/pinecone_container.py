import os

import numpy as np
from keras.src.applications.resnet import preprocess_input
from pinecone import Pinecone


def image_to_vector(img, model):
    resized_img = np.resize(img, (224, 224, 3))  # Resize the image to (224, 224)
    preprocessed_img = preprocess_input(
        resized_img[np.newaxis, ...]
    )  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)
    return query_feature[0].tolist()


class PineconeContainer:
    def __init__(self):
        self.pc = Pinecone(api_key=os.environ["API_KEY"], environment=os.environ["ENV"])
        self.index = self.pc.Index(name="bottle-caps")

    def query_database(self, vector):
        result = self.index.query(vector=[vector], top_k=5, namespace="bottle_caps")
        return self.parse_result_query(result)

    def query_with_metadata(self, vector: list[float], metadata: dict):
        result = self.index.query(
            vector=vector,
            filter=metadata,
            top_k=5,
            include_metadata=True,
            namespace="bottle_caps",
        )
        return self.parse_result_query(result)

    def upsert_to_pinecone(self, cap_info):
        self.index.upsert(vectors=[cap_info], namespace="bottle_caps")

    def parse_result_query(self, result_query):
        return result_query["matches"]
