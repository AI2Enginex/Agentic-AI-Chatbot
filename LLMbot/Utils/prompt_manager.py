from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

class PromptTemplates:

    '''
    class for creating prompt Instructions for the LLM
    '''
    
    # creating the input prompt template 
    # as an input to the model
    @classmethod
    def chat_prompt_template(cls):

        try:
            template = """You are a helpful and knowledgeable assistant. Your task is to respond accurately to user queries.

            INSTRUCTIONS:
            1. Provide clear, concise, and accurate responses based on your knowledge.
            2. If you are confident in your answer, respond naturally and completely.
            3. If you are uncertain or lack sufficient information to provide a reliable answer, explicitly state:
            - "I'm not certain about this" or "I don't have enough information"
            - Briefly explain what you're uncertain about
            4. Do not fabricate or guess information. It's better to admit uncertainty than provide incorrect data.
            5. Structure your response clearly with proper formatting.

            RESPONSE FORMAT:
            {format_instructions}

            USER QUERY:
            {query}

            Please provide a well-structured response following the format instructions above."""

            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
        
    @classmethod
    def comma_seperated_prompt_template(cls):

        try:
            template = """You are a helpful assistant specialized in generating structured lists.

            INSTRUCTIONS:
            1. Generate a comma-separated list of items as per the user's query.
            2. Ensure each item is relevant and appropriate.
            3. If some items are uncertain or not well-established, mark them with a [?] prefix.
            4. If you cannot generate a complete list, explain which items you're confident about.
            5. Keep items concise but descriptive.

            RESPONSE FORMAT:
            {format_instructions}

            USER QUERY:
            {query}

            Please provide the list following the format instructions above."""

            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
    
    @classmethod
    def confidence_aware_prompt_template(cls):
        """
        Template that explicitly asks the LLM to indicate confidence level.
        Use this for queries where confidence scoring is important.
        """
        try:
            template = """You are a helpful assistant. Respond to the user's query with explicit confidence indication.

            INSTRUCTIONS:
            1. Provide your answer to the query.
            2. Rate your confidence in the answer on a scale: HIGH, MEDIUM, or LOW
            - HIGH: You are very confident in this information
            - MEDIUM: You have reasonable confidence but some uncertainty exists
            - LOW: You are uncertain or lack sufficient information
            3. If confidence is MEDIUM or LOW, briefly explain why you lack complete certainty.
            4. Never provide false or fabricated information. Uncertainty is acceptable.

            RESPONSE FORMAT:
            {format_instructions}

            Confidence Level: [HIGH/MEDIUM/LOW]
            Explanation (if needed): [Brief explanation of uncertainty]

            USER QUERY:
            {query}

            Please provide a complete response with confidence level."""

            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
    
    @classmethod
    def research_hint_prompt_template(cls):
        """
        Template that asks the LLM to indicate if web research would be helpful.
        Use this to get signals about when web search should be triggered.
        """
        try:
            template = """You are a helpful assistant that provides transparent responses about knowledge limitations.

            INSTRUCTIONS:
            1. Answer the user's query based on your training knowledge.
            2. If external/recent information or web search would significantly improve the answer, explicitly state:
            - "Web research could help with: [specific aspect]"
            3. Provide your best answer even if it would benefit from additional research.
            4. Be honest about knowledge cutoffs or areas requiring current information.
           

            RESPONSE FORMAT:
            {format_instructions}

            Web Search Suggestion: [Yes/No/Partial]
            (If Yes/Partial, specify what would be helpful)

            USER QUERY:
            {query}

            Please provide your response with the format above."""

            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
    
    @classmethod
    def structured_response_prompt_template(cls):
        """
        Template for getting structured responses with confidence and reasoning.
        Ideal for complex queries requiring detailed analysis.
        """
        try:
            template = """You are an expert assistant providing detailed, well-reasoned responses.

                INSTRUCTIONS:
                1. Break down the query into key components.
                2. Address each component clearly and logically.
                3. Provide reasoning for your conclusions.
                4. Indicate confidence level for each major claim.
                5. If information is outdated or uncertain, flag it appropriately.

                RESPONSE STRUCTURE:
                - Main Answer: [Clear, direct response]
                - Key Points: [Important details and supporting information]
                - Confidence Level: [HIGH/MEDIUM/LOW for overall response]
                - Limitations: [Any caveats, unknowns, or areas needing verification]

                RESPONSE FORMAT:
                {format_instructions}

                USER QUERY:
                {query}

                Please provide a comprehensive structured response."""

            system_prompt = SystemMessagePromptTemplate.from_template(template)
            return ChatPromptTemplate.from_messages([system_prompt])

        except Exception as e:
            return e
        
    @classmethod
    def refinement_prompt_template(cls, query: str,output: str, web_results: str):

        """
        Template for refining an existing answer using new web search results.
        Use this when you want the LLM to improve its previous response based on new information.
        """

        try:
            refinement_ = f"""
            Original Question: {query}

            Your Previous Response: {output}

            Current Web Search Results:
            {web_results}

            Instructions:
            1. Review your previous response and the web search results
            2. Incorporate the latest information from the search results
            3. Provide an improved, comprehensive answer that combines both your knowledge and the search results
            4. Focus on accuracy and relevance
            5. Clearly indicate which information comes from web search (e.g., "According to recent sources..." or "Latest information shows...")

            Please provide an improved response:
            """
            return refinement_
        except Exception as e:
            return e