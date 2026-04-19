from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from langgraph.graph import StateGraph
import google.generativeai as genai  
from langchain_google_genai import ChatGoogleGenerativeAI
# Load variables from .env
load_dotenv()

# Loading the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)


class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = genai.GenerationConfig(
            temperature=1.0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.llm=ChatGoogleGenerativeAI(temperature=0.85,model="gemini-3-flash-preview", google_api_key=api_key,top_p=1.0,
            top_k=32,
            max_output_tokens=15000)


# Declaring LLMOutputOnlyString Model class
# this is how the LLM is 
# expected to output only a string 
class LLMOutputOnlyString(BaseModel):

    # the LLM is expected to output a strings
    content: str = Field(description="The output can be a plain string")

class QueryStateForString(BaseModel):
    query: str
    output: Optional[str] = None # the LLM is expected to output only as a string.
    confidence_score: float = 0.0  # Confidence in LLM response (0-1)
    web_search_results: Optional[str] = None  # Web search results if needed
    needs_web_search: bool = False  # Flag to trigger web search

# Declaring LLMOutput Model class
# this is how the LLM is expected to 
# output the response
class LLMOutput(BaseModel):

    # the LLM is expected to output a strings
    content: Union[str, List[str]] = Field(description="The output can be a plain string or a list of strings")

# Declaring a Final Output class 
# format to be displayed
class QueryState(BaseModel):
    query: str
    output: Optional[Union[str, List[str]]] = None
    human_approval: Optional[str] = None
    next: Optional[str] = None
    rejection_count: int = 0
    # NEW FIELDS:
    confidence_score: float = 0.0  # Confidence in LLM response (0-1)
    web_search_results: Optional[str] = None  # Web search results if needed
    needs_web_search: bool = False  # Flag to trigger web search
