import os
import pinecone

class PineconeContainer:
    def __init__(self):
        self.api_key = os.environ["API_KEY"]
        self.env = os.environ["ENV"]
        pinecone.init(api_key=self.api_key, environment=self.env)
        self.index = pass