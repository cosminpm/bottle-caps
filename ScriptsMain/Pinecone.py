import os
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
        result = self.index.query(vector=[vector], top_k=50, namespace="bottle_caps")
        return self.parse_result_query(result)

    def upsert_to_pinecone(self, vector):
        self.index.upsert(
            vectors=[
                vector
            ],
            namespace='bottle_caps'
        )

    def parse_result_query(self, result_query):
        return result_query['matches']
