import os
from typing import List
import openai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Ensure the API key is loaded from the environment
openai_api_key = os.getenv("OPENAI_API")

class OpenAIEmbedding:
    """
    A wrapper class for generating embeddings using OpenAI's API.
    """

    def __init__(self, model: str = "text-embedding-ada-002", max_retries: int = 3):
        """
        Initializes the OpenAI embedding model.

        Args:
            model (str): The OpenAI embedding model to use. Default is 'text-embedding-ada-002'.
            max_retries (int): Maximum number of retries for API calls. Default is 3.
        """
        self.model = model
        self.max_retries = max_retries
        self._validate_api_key()

    def _validate_api_key(self):
        """
        Validates that the OpenAI API key is set in the environment variables.
        """
        if not openai_api_key:
            raise EnvironmentError("OpenAI API key not found. Please set it in your environment variables.")
        openai.api_key = openai_api_key  # Set the API key for OpenAI

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): Input text to embed.

        Returns:
            List[float]: A list of float values representing the embedding.
        """
        for attempt in range(self.max_retries):
            try:
                response = openai.Embedding.create(input=text, model=self.model)
                return response["data"][0]["embedding"]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error generating embedding. Retrying... ({attempt + 1}/{self.max_retries})")
                else:
                    raise RuntimeError(f"Failed to generate embedding after {self.max_retries} attempts. Error: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts to embed.

        Returns:
            List[List[float]]: A list of embeddings, each corresponding to an input text.
        """
        return [self.get_embedding(text) for text in texts]

def get_embedding_model() -> OpenAIEmbedding:
    """
    Factory function to create an instance of the OpenAIEmbedding class.

    Returns:
        OpenAIEmbedding: An instance of OpenAIEmbedding.
    """
    return OpenAIEmbedding()

if __name__ == "__main__":
    # Example usage
    embedding_model = get_embedding_model()
    test_text = "Llama Index is a great framework for information retrieval."
    embedding = embedding_model.get_embedding(test_text)

    print(f"Generated Embedding (truncated): {embedding[:10]} ...")  # Print the first 10 values
