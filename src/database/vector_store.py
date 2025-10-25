import chromadb
from chromadb.utils import embedding_functions
from src.config.settings import settings

class RecipeVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.AZURE_OPENAI_KEY_EMBEDDING,
            model_name=settings.AZURE_OPENAI_KEY_EMBEDDING_MODEL,
            api_base=settings.AZURE_OPENAI_ENDPOINT,
        )
        self._drop_collection()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name="recipes",
            embedding_function=self.embedding_fn
        )
    
    def _drop_collection(self):
        self.client.delete_collection(name="recipes")

    def add_recipe(self, recipe_id: str, recipe_text: str, metadata: dict):
        self.collection.add(
            ids=[recipe_id],
            documents=[recipe_text],
            metadatas=[metadata]
        )

    def search_recipes(self, query: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )