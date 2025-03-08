import logging
import os
import json
import numpy as np
import requests
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv, set_key
from openai import OpenAI
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.openai_connection")

class OpenAIConnectionError(Exception):
    """Base exception for OpenAI connection errors"""
    pass

class OpenAIConfigurationError(OpenAIConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class OpenAIAPIError(OpenAIConnectionError):
    """Raised when OpenAI API requests fail"""
    pass

class RAGDocument:
    """Class to represent a document in the RAG system"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None
        
    def __str__(self):
        return f"Document: {self.content[:50]}... Metadata: {self.metadata}"

class OpenAIConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._documents = []  # Store RAG documents
        self._embeddings = []  # Store corresponding embeddings

    @property
    def is_llm_provider(self) -> bool:
        return True

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OpenAI configuration from JSON"""
        required_fields = ["model"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")
            
        # Validate model exists (will be checked in detail during configure)
        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")
            
        return config

    def register_actions(self) -> None:
        """Register available OpenAI actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                    ActionParameter("use_rag", False, bool, "Whether to use RAG for enhanced responses")
                ],
                description="Generate text using OpenAI models with optional RAG enhancement"
            ),
            "check-model": Action(
                name="check-model",
                parameters=[
                    ActionParameter("model", True, str, "Model name to check availability")
                ],
                description="Check if a specific model is available"
            ),
            "list-models": Action(
                name="list-models",
                parameters=[],
                description="List all available OpenAI models"
            ),
            "load-rag-data": Action(
                name="load-rag-data",
                parameters=[
                    ActionParameter("url", True, str, "URL to fetch RAG data from"),
                    ActionParameter("risk", True, str, "Risk profile for RAG processing")
                ],
                description="Load data from URL for RAG processing"
            ),
            "search-rag-documents": Action(
                name="search-rag-documents",
                parameters=[
                    ActionParameter("query", True, str, "Query to search for relevant documents"),
                    ActionParameter("top_k", False, int, "Number of top results to return")
                ],
                description="Search RAG documents for relevant information"
            )
        }

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client"""
        if not self._client:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise OpenAIConfigurationError("OpenAI API key not found in environment")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def configure(self) -> bool:
        """Sets up OpenAI API authentication"""
        logger.info("\n🤖 OPENAI API SETUP")

        if self.is_configured():
            logger.info("\nOpenAI API is already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your OpenAI API credentials:")
        logger.info("1. Go to https://platform.openai.com/account/api-keys")
        logger.info("2. Create a new project or open an existing one.")
        logger.info("3. In your project settings, navigate to the API keys section and create a new API key")
        
        api_key = input("\nEnter your OpenAI API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'OPENAI_API_KEY', api_key)
            
            # Validate the API key by trying to list models
            client = OpenAI(api_key=api_key)
            client.models.list()

            logger.info("\n✅ OpenAI API configuration successfully saved!")
            logger.info("Your API key has been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose = False) -> bool:
        """Check if OpenAI API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False

            client = OpenAI(api_key=api_key)
            client.models.list()
            return True
            
        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    def load_rag_data(self, url: str, risk: str, **kwargs) -> bool:
        """Load data from URL for RAG processing"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            data = self.process_data(data, risk)
            
            self._documents = []
            
            if "kols" in data:
                for kol in data["kols"]:
                    kol_doc = self._create_kol_document(kol)
                    self._documents.append(kol_doc)
                    
                    if "tweets" in kol:
                        for tweet in kol["tweets"]:
                            tweet_doc = self._create_tweet_document(tweet, kol)
                            self._documents.append(tweet_doc)
            
            self._generate_embeddings()
            
            logger.info(f"Successfully loaded {len(self._documents)} documents for RAG")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RAG data: {e}")
            raise OpenAIAPIError(f"Failed to load RAG data: {e}")
    
    def process_data(self, data, risk):
        filtered_kols = []
    
        for kol in data["kols"]:
            if kol["riskRecommendation"].lower() == risk:
                filtered_tweets = [tweet for tweet in kol["tweets"] if not tweet["expired"]]
                if filtered_tweets:
                    filtered_kols.append({**kol, "tweets": filtered_tweets})
        
        return {"kols": filtered_kols}
        

    def _create_kol_document(self, kol: Dict[str, Any]) -> RAGDocument:
        """Create a document from KOL data"""
        content = (
            f"KOL Name: {kol.get('name')}\n"
            f"Username: {kol.get('username')}\n"
            f"Twitter Followers: {kol.get('followersTwitter')}\n"
            f"KOL Followers: {kol.get('followersKOL')}\n"
            f"Risk Recommendation: {kol.get('riskRecommendation')}\n"
            f"Average Profit (Daily): {kol.get('avgProfitD')}%\n"
            f"Rank (Followers): {kol.get('rankFollowersKOL')}\n"
            f"Rank (Profit): {kol.get('rankAvgProfitD')}\n"
        )
        
        metadata = {
            "id": kol.get("id"),
            "type": "kol",
            "name": kol.get("name"),
            "username": kol.get("username"),
            "risk_level": kol.get("riskRecommendation")
        }
        
        return RAGDocument(content, metadata)

    def _create_tweet_document(self, tweet: Dict[str, Any], kol: Dict[str, Any]) -> RAGDocument:
        """Create a document from Tweet data"""
        token = tweet.get("token", {})
        
        content = (
            f"Tweet from {kol.get('name')} (@{kol.get('username')}):\n"
            f"Content: {tweet.get('content')}\n"
            f"Signal: {tweet.get('signal')}\n"
            f"Risk Level: {tweet.get('risk')}\n"
            f"Token: {token.get('name')} (${token.get('symbol')})\n"
            f"Token Price Change (24h): {token.get('priceChange24H')}\n"
            f"Token Tags: {', '.join(token.get('tags', []))}\n"
        )
        
        metadata = {
            "id": tweet.get("id"),
            "type": "tweet",
            "kol_id": kol.get("id"),
            "kol_name": kol.get("name"),
            "token_symbol": token.get("symbol"),
            "signal": tweet.get("signal"),
            "risk_level": tweet.get("risk")
        }
        
        return RAGDocument(content, metadata)

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all documents using text-embedding-ada-002"""
        client = self._get_client()
        
        # Process documents in batches to avoid API limits
        batch_size = 20
        for i in range(0, len(self._documents), batch_size):
            batch = self._documents[i:i+batch_size]
            texts = [doc.content for doc in batch]
            
            try:
                response = client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )
                
                # Store embeddings in the corresponding documents
                for j, embedding_data in enumerate(response.data):
                    self._documents[i+j].embedding = embedding_data.embedding
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size}: {e}")
                raise OpenAIAPIError(f"Failed to generate embeddings: {e}")

    def search_rag_documents(self, query: str, top_k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """Search for relevant documents using cosine similarity"""
        if not self._documents or not all(doc.embedding for doc in self._documents):
            raise OpenAIAPIError("No documents with embeddings available. Load RAG data first.")
        
        # Generate query embedding
        client = self._get_client()
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate similarities
        similarities = []
        for doc in self._documents:
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((similarity, doc))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top_k results
        results = []
        for similarity, doc in similarities[:top_k]:
            results.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "similarity": similarity
            })
            
        return results

    def _cosine_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)

    def generate_text(self, prompt: str, system_prompt: str, model: str = None, use_rag: bool = False, **kwargs) -> str:
        """Generate text using OpenAI models with optional RAG enhancement"""
        try:
            client = self._get_client()
            
            # Use configured model if none provided
            if not model:
                model = self.config["model"]
                
            # If RAG is enabled, retrieve relevant documents and enhance the prompt
            enhanced_prompt = prompt
            if use_rag and self._documents:
                try:
                    # Search for relevant documents
                    relevant_docs = self.search_rag_documents(prompt, top_k=3)
                    
                    if relevant_docs:
                        # Format the context from retrieved documents
                        context = "\n\n".join([doc["content"] for doc in relevant_docs])
                        
                        # Enhance the prompt with retrieved context
                        enhanced_prompt = (
                            f"I have some specific information that might help answer the query:\n\n"
                            f"{context}\n\n"
                            f"Using this information where relevant, please respond to the following query:\n{prompt}"
                        )
                        
                        logger.info(f"Enhanced prompt with {len(relevant_docs)} relevant documents")
                    
                except Exception as e:
                    logger.warning(f"RAG enhancement failed, using original prompt: {e}")
            
            # Generate completion with potentially enhanced prompt
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt},
                ],
            )

            return completion.choices[0].message.content
            
        except Exception as e:
            raise OpenAIAPIError(f"Text generation failed: {e}")

    def check_model(self, model, **kwargs):
        try:
            client = self._get_client()
            try:
                client.models.retrieve(model=model)
                # If we get here, the model exists
                return True
            except Exception:
                return False
        except Exception as e:
            raise OpenAIAPIError(e)

    def list_models(self, **kwargs) -> None:
        """List all available OpenAI models"""
        try:
            client = self._get_client()
            response = client.models.list().data
            
            fine_tuned_models = [
                model for model in response 
                if model.owned_by in ["organization", "user", "organization-owner"]
            ]

            logger.info("\nGPT MODELS:")
            logger.info("1. gpt-3.5-turbo")
            logger.info("2. gpt-4")
            logger.info("3. gpt-4-turbo")
            logger.info("4. gpt-4o")
            logger.info("5. gpt-4o-mini")
            
            if fine_tuned_models:
                logger.info("\nFINE-TUNED MODELS:")
                for i, model in enumerate(fine_tuned_models):
                    logger.info(f"{i+1}. {model.id}")
                    
        except Exception as e:
            raise OpenAIAPIError(f"Listing models failed: {e}")
    
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Twitter action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)