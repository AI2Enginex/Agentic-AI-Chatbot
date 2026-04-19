import re
from typing import Tuple

class ConfidenceChecker:
    """
    Evaluates LLM response confidence to determine if web search is needed
    """
    
    # Keywords indicating low confidence
    UNCERTAIN_KEYWORDS = [
        "i don't know",
        "i'm not sure",
        "i cannot find",
        "i don't have",
        "not available",
        "unclear",
        "uncertain",
        "may vary",
        "depends on",
        "no information",
        "unable to provide",
        "don't have access",
        "don't have real-time",
        "not enough information",
        "cannot tell you",
        "unable to determine",
        "no data",
        "not provided",
        "cannot answer",
        "lack of information",
        "beyond my knowledge",
        "no knowledge",
        "not available in my",
        "my knowledge cutoff",
        "based on my last training",
        "knowledge is based on",
        "training update and does not",
        "does not include current",
        "does not include live",
        "consult a real-time",
        "financial news source",
        "stock market tracking",
        "news aggregator",
        "up-to-the-minute",
        "current events or live",
        "need to consult",
        "you would need to",
        "i apologize",
        "i'm unable to",
    ]
    
    # Keywords indicating refusal or strong limitation
    REFUSAL_KEYWORDS = [
        "i can't",
        "i cannot",
        "not able to",
        "unable to",
        "not possible",
        "cannot help",
        "cannot provide",
        "cannot access",
        "don't have access",
        "real-time",
        "live data",
        "current data",
        "cannot provide real-time",
        "apologize, but i cannot"
    ]
    
    # Keywords indicating the query needs current/live data
    CURRENT_DATA_KEYWORDS = [
        "today",
        "yesterday",
        "tomorrow",
        "current",
        "latest",
        "recent",
        "now",
        "live",
        "real-time",
        "this week",
        "this month",
        "this year",
        "2024",
        "2025",
        "2026",
        "top market news",
        "market news",
        "stock",
        "nse",
        "bse",
        "news",
        "latest news",
        "breaking"
    ]
    
    @classmethod
    def check_response_confidence(cls, response: str) -> Tuple[float, bool]:
        """
        Check confidence score of LLM response
        
        Args:
            response: LLM generated response
            
        Returns:
            Tuple of (confidence_score: float 0-1, needs_web_search: bool)
        """
        response_lower = response.lower()
        
        # Check for uncertain keywords
        uncertain_count = sum(1 for keyword in cls.UNCERTAIN_KEYWORDS 
                             if keyword in response_lower)
        
        # Check for refusal keywords
        refusal_count = sum(1 for keyword in cls.REFUSAL_KEYWORDS 
                           if keyword in response_lower)
        
        # Check for current/live data keywords
        needs_current_data = sum(1 for keyword in cls.CURRENT_DATA_KEYWORDS 
                                if keyword in response_lower)
        
        # Calculate confidence score
        confidence_score = 1.0
        needs_web_search = False
        
        if refusal_count > 0:
            confidence_score = 0.05
            needs_web_search = True
            print(f"[Confidence Check] Strong refusal detected: confidence={confidence_score:.2f}, web_search={needs_web_search}")
        elif uncertain_count > 0:
            confidence_score = max(0.1, 0.4 - (uncertain_count * 0.15))
            needs_web_search = True
            print(f"[Confidence Check] Uncertainty detected ({uncertain_count} keywords): confidence={confidence_score:.2f}, web_search={needs_web_search}")
        
        # Boost web search trigger for current/real-time data requests
        if needs_current_data > 0 and confidence_score < 0.6:
            needs_web_search = True
            print(f"[Confidence Check] Real-time/current data requested: forcing web_search=True")
        
        # Minimum response length check - very short responses are often uncertain
        response_words = response.split()
        if len(response_words) < 15 and refusal_count > 0:
            confidence_score = min(confidence_score, 0.05)
            needs_web_search = True
            print(f"[Confidence Check] Short refusal response ({len(response_words)} words): confidence={confidence_score:.2f}, web_search={needs_web_search}")
        
        return confidence_score, needs_web_search
    
    @staticmethod
    def is_verbose_apology(response: str) -> bool:
        """
        Check if response is a verbose apology/refusal that should be suppressed
        Catches various patterns of verbose refusals
        """
        response_lower = response.lower()
        
        # Direct apology/refusal patterns
        apology_patterns = [
            "i apologize",
            "i cannot provide",
            "i'm unable to",
            "i do not have",
            "i cannot provide real-time",
            "i'm unable to",
            "knowledge is based on",
            "knowledge base is",
            "my knowledge cutoff",
            "consult a real-time",
            "you would need to consult",
            "my knowledge base",
            "as an ai",
            "cannot provide accurate",
            "do not have the ability",
            "do not have access",
            "cannot describe",
            "cannot predict",
            "lack the ability",
            "beyond my capability"
        ]
        
        # Check for direct patterns
        if any(pattern in response_lower for pattern in apology_patterns):
            return True
        
        # Advanced detection: Check for multiple refusal indicators
        refusal_indicators = ["cannot", "unable", "apologize", "do not have", "i don't have"]
        knowledge_indicators = ["knowledge", "training", "updated", "access", "real-time", "live", "current"]
        
        refusal_count = sum(1 for indicator in refusal_indicators if indicator in response_lower)
        knowledge_count = sum(1 for indicator in knowledge_indicators if indicator in response_lower)
        
        # If 2+ refusals combined with 2+ knowledge limitations, likely verbose
        if refusal_count >= 2 and knowledge_count >= 2:
            return True
        
        return False
    
    @staticmethod
    def clean_verbose_response(response: str):
        """
        Remove verbose apologies from response if web search is being triggered
        Replace with a brief note
        """
        if ConfidenceChecker.is_verbose_apology(response):
            return "⏳ **Searching the web for current information...**"
        return response