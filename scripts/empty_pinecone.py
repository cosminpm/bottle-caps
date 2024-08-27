from dotenv import load_dotenv

from app.services.identify.pinecone_container import PineconeContainer


def main():
    load_dotenv()
    pinecone_container: PineconeContainer = PineconeContainer()
    pinecone_container.empty_index()


if __name__ == '__main__':
    main()
