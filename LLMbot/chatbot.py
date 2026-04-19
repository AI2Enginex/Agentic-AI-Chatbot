from Utils.utils import ChatGoogleGENAI, QueryState
from Utils.output_parser import PydanticOutputs
from langgraph.graph import StateGraph, END
from Utils.prompt_manager import PromptTemplates
from Utils.web_search_tool import WebSearchTool
from Utils.confidence_checker import ConfidenceChecker
    
# declaring a class HumanInTheLoop
# for demonstrating working of HITL 
class HumanInTheLoop(ChatGoogleGENAI): 

    def __init__(self, state_cls,output_state):
        super().__init__()
        self.state_cls = state_cls  # accepts multiple states
        self.opstate = output_state
        self.web_search_tool = WebSearchTool()  # Initialize web search tool

    # generate output Node
    def generate_output(self, state):
        try:
            print(f"\nProcessing query: '{state.query}'")
            
            inputs = {
                "query": state.query,
                "format_instructions": PydanticOutputs.DictOutputParser(method=self.opstate).get_format_instructions()
            }

            messages = PromptTemplates.chat_prompt_template().format(**inputs)
            response = self.llm.invoke(messages)
            
            # Handle response content - can be string or list of content blocks
            response_content = response.content
            if isinstance(response_content, list):
                # Extract text from content blocks
                print(f"[Response Format] Extracting text from content blocks")
                text_parts = []
                for block in response_content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_content = "".join(text_parts)
            
            result = PydanticOutputs.DictOutputParser(method=self.opstate).parse(response_content)

            # Calculate confidence score and determine if web search is needed
            confidence_score, needs_web_search = ConfidenceChecker.check_response_confidence(result.content)
            
            print(f"LLM Response: {result.content}...")
            print(f"Confidence Score: {confidence_score:.2f} | Needs Web Search: {needs_web_search}")

            # Build state dict with only required attributes
            state_dict = {
                "query": state.query,
                "output": result.content,
                "confidence_score": confidence_score,
                "needs_web_search": needs_web_search,
                "web_search_results": None,
            }
            
            # Add optional attributes if they exist on the state class
            if hasattr(state, 'human_approval'):
                state_dict['human_approval'] = None
            if hasattr(state, 'next'):
                state_dict['next'] = None
            if hasattr(state, 'rejection_count'):
                state_dict['rejection_count'] = 0
            
            return self.state_cls(**state_dict)
        except Exception as e:
            print(f'Error in generate_output: {e}')
            import traceback
            traceback.print_exc()
            return state

   # Web search node
    def perform_web_search(self, state):
        """
        Perform web search if LLM confidence is low
        """
        try:
            print(f"\nTriggering web search for query: '{state.query}'")
            print(f"LLM Confidence: {state.confidence_score:.2f}")
            
            # Perform the search - fetch more results for better coverage
            search_results = self.web_search_tool.search(state.query, num_results=5)
            
            print(f"Search returned {len(search_results)} results")
            
            # Format results
            formatted_results = self.web_search_tool.format_search_results(search_results)
            
            # Return state with search results
            # Build state dict with only attributes that exist in the state class
            state_dict = {
                "query": state.query,
                "output": state.output,
                "web_search_results": formatted_results,
                "confidence_score": state.confidence_score,
                "needs_web_search": False,
            }
            
            # Add optional attributes if they exist on the state
            if hasattr(state, 'human_approval'):
                state_dict['human_approval'] = state.human_approval
            if hasattr(state, 'next'):
                state_dict['next'] = state.next
            if hasattr(state, 'rejection_count'):
                state_dict['rejection_count'] = state.rejection_count
            
            return self.state_cls(**state_dict)
        except Exception as e:
            print(f'Error in web search: {e}')
            import traceback
            traceback.print_exc()
            return state
    
    def refine_with_search(self, state):
        """
        Pass web search results back to LLM for refinement
        LLM combines its original response with web search data
        """
        try:
            print(f"\nRefining LLM response with web search data...")
            
            # Create a refinement prompt that asks LLM to incorporate search results
            refinement_prompt = PromptTemplates.refinement_prompt_template(
                query=state.query, 
                output=state.output, 
                web_results=state.web_search_results)
            # refinement_prompt = f"""
            # Original Question: {state.query}

            # Your Previous Response: {state.output}

            # Current Web Search Results:
            # {state.web_search_results}

            # Instructions:
            # 1. Review your previous response and the web search results
            # 2. Incorporate the latest information from the search results
            # 3. Provide an improved, comprehensive answer that combines both your knowledge and the search results
            # 4. Focus on accuracy and relevance
            # 5. Clearly indicate which information comes from web search (e.g., "According to recent sources..." or "Latest information shows...")

            # Please provide an improved response:
            # """
            
            # Invoke LLM with refinement prompt
            response = self.llm.invoke(refinement_prompt)
            
            # Handle response content - can be string or list of content blocks
            response_content = response.content
            if isinstance(response_content, list):
                print(f"Extracting refined response from content blocks")
                text_parts = []
                for block in response_content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_content = "".join(text_parts)
            
            refined_output = response_content.strip()
            
            print(f"Response refined with web search data")
            
            # Build state dict with refined output
            state_dict = {
                "query": state.query,
                "output": refined_output,
                "web_search_results": state.web_search_results,
                "confidence_score": state.confidence_score,
                "needs_web_search": False,
            }
            
            # Add optional attributes if they exist on the state
            if hasattr(state, 'human_approval'):
                state_dict['human_approval'] = state.human_approval
            if hasattr(state, 'next'):
                state_dict['next'] = state.next
            if hasattr(state, 'rejection_count'):
                state_dict['rejection_count'] = state.rejection_count
            
            return self.state_cls(**state_dict)
        except Exception as e:
            print(f'Error in refine_with_search: {e}')
            import traceback
            traceback.print_exc()
            return state
    
    def final_output(self, state):
        """
        Return the final refined response
        If LLM was refined with web search, output already includes search data
        If no web search was performed, return the original LLM response
        """
        try:
            output_text = state.output or ""
            is_verbose = ConfidenceChecker.is_verbose_apology(output_text)
            
            if state.web_search_results and "No results found" not in state.web_search_results:
                # Web search was performed and results were incorporated by LLM
                print(f"\nFinal response (refined with web search):")
                if is_verbose:
                    print(f"(Verbose apology suppressed)")
                print(f"Response ready for display")
                
            elif state.web_search_results:
                # Web search found no results, but LLM was still refined
                print(f"\nWeb search found no results, using original response")
                if is_verbose:
                    state.output = "I don't have current information on this topic. Please try refining your query or searching with different terms."
                else:
                    state.output = output_text
                    
            else:
                # No web search was triggered
                if is_verbose and state.confidence_score < 0.5:
                    # LLM gave verbose apology and confidence is low
                    print(f"\nVerbose apology detected with low confidence")
                    state.output = "I don't have reliable information about this. Please try another search or rephrase your query."
                elif is_verbose:
                    # Verbose but confidence was high enough to not trigger web search
                    print(f"\n! Verbose response but confidence threshold not met")
                    print(f"Confidence: {state.confidence_score:.2f}")
                    state.output = ConfidenceChecker.clean_verbose_response(output_text)
                else:
                    print(f"\nUsing LLM response (confidence: {state.confidence_score:.2f})")
            
            return state
        except Exception as e:
            print(f'Error in final_output: {e}')
            import traceback
            traceback.print_exc()
            return state
        
# Define a new class that builds and runs a graph for human-in-the-loop processing
class GraphForHumanInTheLoop(HumanInTheLoop):
    
    # When this class is created, we take two arguments: the state and the output state
    def __init__(self, state: None, opstate: None):
        # We initialize the parent HumanInTheLoop class with the given state and output state
        super().__init__(state_cls=state, output_state=opstate)
        
    # This method builds the actual graph logic
    def build_graph(self):
        try:
            builder = StateGraph(self.state_cls)
            
            # Add nodes
            builder.add_node("generate_output", self.generate_output)
            builder.add_node("web_search", self.perform_web_search)
            builder.add_node("refine_response", self.refine_with_search)
            builder.add_node("final_decision", self.final_output)
            
            # Set entry point
            builder.set_entry_point("generate_output")
            
            # Conditional logic: only perform search if needs_web_search is True
            def should_perform_search(state):
                return "web_search" if state.needs_web_search else "final_decision"
            
            # Add conditional edges from generate_output
            builder.add_conditional_edges(
                "generate_output",
                should_perform_search,
                {"web_search": "web_search", "final_decision": "final_decision"}
            )
            
            # Connect web_search to refine_response (pass results to LLM for refinement)
            builder.add_edge("web_search", "refine_response")
            
            # Connect refine_response to final_decision
            builder.add_edge("refine_response", "final_decision")
            
            # Connect final_decision to END
            builder.add_edge("final_decision", END)
            
            return builder
            
        except Exception as e:
            return e
        
    # This method actually runs the graph with a user's query
    def execuete_graph(self, user_query: str):
        try:
            # First, build the graph structure
            graph_builder = self.build_graph()

            # Compile the graph so it can be executed
            execuetor = graph_builder.compile()
            
            # Create the starting state using the user's input
            initial_state = QueryState(query=user_query)

            # Run the graph and get the result
            result = execuetor.invoke(initial_state)

            # Return the final result after the graph finishes
            return result

        # If anything goes wrong while running the graph, print the error
        except Exception as e:
            print(f"Fatal error running LangGraph: {e}")


if __name__ == "__main__":
    pass