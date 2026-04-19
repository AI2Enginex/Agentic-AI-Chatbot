import streamlit as st

from Utils.utils import QueryStateForString, LLMOutputOnlyString 
from chatbot import GraphForHumanInTheLoop


# =========================================================
# Execution class
# Handles chatbot logic, stores user input, rejection count, and model interactions
# =========================================================
class Execution:
    def __init__(self, user_input):
        """
        Initialize the chatbot execution session.
        Args:
            user_input (str): The user's input question or message.
        """
        self.user_input = user_input                # Store user input
        self.rejection_count = 0                    # Count number of times user rejects the output
        self.human_approval = None                  # Track whether user approved the response
        # Initialize the chatbot graph with predefined state and output configuration
        self.g = GraphForHumanInTheLoop(
            state=QueryStateForString,
            opstate=LLMOutputOnlyString
        )

    def get_chatbot_output(self):
        """
        Run the chatbot model and return its response.
        Returns:
            dict: Output from chatbot or error message.
        """
        try:
            result = self.g.execuete_graph(user_query=self.user_input)
            # Convert Pydantic model to dict if necessary
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"output": str(result)}
        except Exception as e:
            print(f"Chatbot Error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": f"Error: {str(e)}"}


# =========================================================
# HITLChatbotApp class
# Manages the Streamlit UI and interaction flow
# =========================================================
class HITLChatbotApp:
    def __init__(self):
        """
        Initialize Streamlit session state variables
        to maintain state between user interactions.
        """
        # Store the chatbot execution object
        if 'execution' not in st.session_state:
            st.session_state.execution = None

        # Store the chatbot's latest generated response
        if 'chatbot_result' not in st.session_state:
            st.session_state.chatbot_result = None

        # Store the final approved answer
        if 'final_output' not in st.session_state:
            st.session_state.final_output = None

    def initialize_execution(self, user_input):
        """
        Create a new Execution instance for the given user input
        and fetch the chatbot's first response.
        """
        st.session_state.execution = Execution(user_input=user_input)
        st.session_state.chatbot_result = st.session_state.execution.get_chatbot_output()
        st.session_state.final_output = None  # Reset final output if user starts new query

    def show_chatbot_response(self):
        """
        Display the chatbot's current response and show
        Accept/Reject buttons for user feedback.
        """
        st.subheader("Chatbot Response:")
        
        # Safely extract output from result
        if isinstance(st.session_state.chatbot_result, dict):
            output = st.session_state.chatbot_result.get('output', 'No response available')
        else:
            output = getattr(st.session_state.chatbot_result, 'output', 'No response available')
        
        # Display with error indication if needed
        if output.startswith('Error:'):
            st.error(output)
        else:
            st.markdown(output, unsafe_allow_html=True)

        # Display two buttons side by side (Accept / Reject)
        col1, col2 = st.columns(2)

        # --- Accept Button ---
        with col1:
            # Use a unique key to prevent Streamlit from reusing the button state
            if st.button("Accept", key="accept_btn"):
                self.handle_accept()

        # --- Reject Button ---
        with col2:
            if st.button("Reject", key="reject_btn"):
                self.handle_reject()

    def handle_accept(self):
        """
        Handle when the user accepts the chatbot response.
        Save the final result in session_state for display.
        """
        st.session_state.execution.human_approval = 'yes'  # Mark approval

        # Safely extract output from result
        if isinstance(st.session_state.chatbot_result, dict):
            reply = st.session_state.chatbot_result.get('output', 'No response')
        else:
            reply = getattr(st.session_state.chatbot_result, 'output', 'No response')

        # Store accepted output details
        st.session_state.final_output = {
            "query": st.session_state.execution.user_input,
            "reply": reply,
            "total_rejections": st.session_state.execution.rejection_count
        }

    def handle_reject(self):
        """
        Handle when the user rejects the chatbot response.
        Increment rejection count and regenerate a new response.
        """
        # Increase the count of rejections
        st.session_state.execution.rejection_count += 1

        # Get a new chatbot response for the same question
        st.session_state.chatbot_result = st.session_state.execution.get_chatbot_output()

        # Force Streamlit to refresh UI and show the new response
        st.rerun()

    def show_final_output(self):
        """
        Display the final approved chatbot response to the user.
        """
        st.success("Final approved response:")
        st.write(st.session_state.final_output['reply'])

    def run(self):
        """
        Main Streamlit UI loop that controls user input,
        chatbot execution, and feedback interactions.
        """
        # --- App Title ---
        st.title("Human-in-the-Loop Chatbot")

        # --- User Input Box ---
        user_input = st.text_input("Enter your question:")

        # --- Submit Button ---
        if st.button("Submit", key="submit_btn") and user_input:
            self.initialize_execution(user_input)

        # --- Show Chatbot Response if available ---
        if st.session_state.chatbot_result and not st.session_state.final_output:
            self.show_chatbot_response()

        # --- Show Final Output if approved ---
        if st.session_state.final_output:
            self.show_final_output()


# =========================================================
# Run the Streamlit Application
# =========================================================
if __name__ == "__main__":
    app = HITLChatbotApp()
    app.run()
