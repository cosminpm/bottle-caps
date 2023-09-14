from ScriptsMain.Pinecone import PineconeContainer

pinecone_container: PineconeContainer = PineconeContainer()

if __name__ == '__main__':
    result = pinecone_container.query_with_metadata(
        metadata={'name': {"$eq": 'test_cap'}}
    )
    print(result)
