import os
from typing import Dict

import pinecone
from keras.applications.resnet import preprocess_input
import numpy as np


def image_to_vector(img, model):
    resized_img = np.resize(img, (224, 224, 3))  # Resize the image to (224, 224)
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)
    return query_feature[0].tolist()


class PineconeContainer:
    def __init__(self):
        self.api_key = os.environ["API_KEY"]
        self.env = os.environ["ENV"]
        pinecone.init(api_key=self.api_key, environment=self.env)
        self.index = pinecone.Index(index_name='bottle-caps')

    def query_database(self, vector):
        result = self.index.query(vector=[vector], top_k=5, namespace="bottle_caps")
        return self.parse_result_query(result)

    def query_with_metadata(self, metadata: Dict):
        return self.index.query(
            vector=[0.1] * 256,
            filter=metadata,
            top_k=5,
            include_metadata=True,
            namespace='bottle_caps'
        )

    def upsert_to_pinecone(self, cap_info):
        self.index.upsert(
            vectors=[
                cap_info
            ],
            namespace='bottle_caps'
        )

    def parse_result_query(self, result_query):
        return result_query['matches']
