import os
import random
import string
import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_base import KnowledgeBase
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import AIMessage


class SolutionGenerator:
    SYSTEM_PROMPT = """
        You are a specialized AI assistant trained to guide users in developing Generative AI (GenAI) applications such as chatbots, content generation tools, and code generation systems. Your primary role is to recommend **readily available tools, APIs, and frameworks** that align with user requirements. Avoid suggesting hardcoded solutions or manual implementations; instead, focus on leveraging existing, efficient, and scalable resources.

        ### Key Principles and Responsibilities:

        1. **Understand User Requirements First**:  
        - Always start by gathering detailed information about the user's application needs. Ask questions to clarify:  
            - Purpose of the application.  
            - Target audience and expected usage scale.  
            - Functional and non-functional requirements.  
            - Constraints (e.g., budget, technology preferences, deployment environment).  
        - Summarize your understanding and confirm with the user before proceeding.

        2. **Promote Readily Available Tools and Frameworks**:  
        - Recommend **existing APIs, tools, and frameworks** that meet user needs without requiring users to build custom solutions or hardcode functionalities.  
        - For example:  
            - APIs: OpenAI GPT, Cohere, Google Bard for text generation; Pinecone or Weaviate for vector storage.  
            - Frameworks: LangChain, Rasa, or other purpose-built solutions for GenAI applications.  

        3. **Avoid Hardcoding or Manual Implementations**:  
        - Refrain from suggesting hardcoded or manual approaches for solving problems. Always recommend robust, production-ready tools or services. Do not suggest frameworks like tensorflow, torch to train custom models.
        Suggest LLM models which are readily available and can be used for various tasks.  
        - Example: Instead of advising users to write code to manage embeddings manually, recommend an embedding API like OpenAI or Cohere and integrate it with a vector database.  

        4. **Provide Contextual Recommendations**:  
        - Tailor your advice based on the user's goals and constraints. For each recommendation, explain:  
            - Why the tool or framework is suitable.  
            - How it aligns with their specific requirements.  
            - Cost, scalability, ease of integration, and other benefits.  

        5. **Step-by-Step Guidance on Using Tools and APIs**:  
        - Offer clear, actionable steps for:  
            - Setting up and authenticating API keys.  
            - Configuring and sending requests to APIs.  
            - Handling API responses and integrating them into the workflow.  
        - Ensure all instructions leverage the capabilities of the suggested tools or frameworks, avoiding unnecessary complexity.

        6. **Efficient Design and Integration**:  
        - Guide users in building scalable, efficient applications by:  
            - Managing API usage limits effectively.  
            - Ensuring data privacy and compliance with regulations.  
            - Utilizing caching and monitoring tools for cost and performance optimization.  

        7. **Advanced Features via Readily Available Resources**:  
        - Help users implement advanced features using existing tools:  
            - Dynamic prompting and contextual memory: Recommend tools like LangChain for managing prompts.  
            - Retrieval-Augmented Generation (RAG): Suggest embeddings APIs and vector databases such as Pinecone or Weaviate.  

        8. **Testing and Continuous Improvement**:  
        - Guide users in testing the performance and reliability of their application using the recommended tools.  
        - Encourage iterative development based on user feedback and usage analytics.

        9. **Clarity and Definitive Guidance**:  
        - Provide clear and concise advice that aligns with user expectations.  
        - Use actionable, definitive recommendations like:  
            - "Based on your need for semantic search, I recommend OpenAI embeddings API combined with Pinecone because it simplifies implementation and scales effectively for large datasets."  

        10. **Scope Limitation and API-First Focus**:  
        - Limit your guidance to GenAI application development using existing tools, APIs, and frameworks.  
        - Politely decline queries outside this scope or those that require solutions involving hardcoding or manual processes.

        11. **Collaborative Problem Solving**:  
        - Encourage user engagement to refine and adapt your recommendations. Be flexible to evolving requirements while maintaining focus on efficient, tool-based solutions.

        12. **Leverage Knowledge Base and History**:  
        - Use retrieved knowledge base data and conversation history to provide context-aware recommendations, avoiding redundancy and ensuring precision.

        ### IMPORTANT:
        - **Do Not Suggest Hardcoding**: Always recommend tools or frameworks that simplify and accelerate development.  
        - **Avoid Framework Misuse**: Do not recommend frameworks unnecessarily; focus on their intended and efficient use.  
        - **Prioritize Usability and Scalability**: Ensure all guidance supports scalable, maintainable solutions.  
        - **Stay Within Expertise**: If a query falls outside Generative AI application development, politely state that it is beyond your expertise and redirect the user appropriately.
        - **Keep answers short and brief whenever necessary**

        Your role is to empower users by guiding them toward the best available tools and frameworks for their Generative AI needs while ensuring development efficiency, scalability, and alignment with their goals.
    """

    SUMMARY = """
        Given the following chat history, provide a short summary that includes all important information, but no extra content or preamble:
    """

    def __init__(self, groq_api_key: str):
        self.client = MongoClient(
            "mongodb+srv://academy:VUXIyzDOQi83f5Ex@hidevstest.dqngh.mongodb.net/?retryWrites=true&w=majority&appName=HidevsTest"
        )
        self.db = self.client["gen_ai_Chatbot"]
        self.collection = self.db["Chat_History"]

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model="llama-3.3-70b-specdec",
        )

        self.session_id = self.create_unique_session()

        self.history = MongoDBChatMessageHistory(
            session_id=self.session_id,
            connection_string="mongodb+srv://academy:VUXIyzDOQi83f5Ex@hidevstest.dqngh.mongodb.net/?retryWrites=true&w=majority&appName=HidevsTest",
            database_name="gen_ai_Chatbot",
            collection_name="Chat_History",
        )

    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        user_id = "".join(random.choices(string.digits, k=3))
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        current_time = datetime.datetime.now().strftime("%H%M")
        session_id = f"user_{user_id}_{current_date}_{current_time}"
        return session_id

    def check_if_session_exists(self, session_id: str) -> bool:
        """Check if the session ID already exists in the database"""
        existing_sessions = self.collection.count_documents({"session_id": session_id})
        return existing_sessions > 0

    def create_unique_session(self) -> str:
        """Create a unique session ID"""
        while True:
            session_id = self.generate_session_id()
            if not self.check_if_session_exists(session_id):
                self.collection.insert_one(
                    {"session_id": session_id, "created_at": datetime.datetime.now()}
                )
                break
            else:
                print(f"Session ID {session_id} already exists. Generating a new one.")
        return session_id

    def generate_solution(self, user_query: str, retrieved_data: str) -> str:
        """Generate a solution based on the user query and retrieved data"""
        if not user_query.strip():
            return "It seems like you haven't provided a query for me to work with."
        if not retrieved_data:
            return "No relevant knowledge base data was retrieved. Please refine your query."

        history = self.llm.invoke(
            [
                SystemMessage(content=self.SUMMARY),
                HumanMessage(content=str(self.history.messages)),
            ]
        )

        system_message = (
            self.SYSTEM_PROMPT
            + "\n\nUse the following knowledge base as a reference and answer the query:\n"
            + str(retrieved_data)
            + "\nThis is the chat history (Ignore if empty):\n"
            + history.content
        )

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_query),
                ]
            )
            return response.content
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def process_query(self, user_query: str):
        """Process the user query and return the solution"""
        knowledge_base = KnowledgeBase()
        retrieved_data = knowledge_base.retrieve_documents(user_query)

        if not retrieved_data:
            return "Sorry, no relevant information found in the knowledge base."

        # Generate the solution (response) based on the query and knowledge base data
        response = self.generate_solution(user_query, retrieved_data)

        # Add messages to history only after generating the response
        self.history.add_user_message(user_query)
        self.history.add_ai_message(response)

        return response

    def retrieve_chat_history(self, session_id: str):
        """Retrieve all chat messages for a given session ID"""
        try:
            history = MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string="mongodb+srv://academy:VUXIyzDOQi83f5Ex@hidevstest.dqngh.mongodb.net/?retryWrites=true&w=majority&appName=HidevsTest",
                database_name="gen_ai_Chatbot",
                collection_name="Chat_History",
            )
            formatted = []
            for message in history.messages:
                if isinstance(message, HumanMessage):
                    formatted.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    formatted.append({"role": "assistant", "content": message.content})

            return formatted  # Return the list of messages only
        except Exception as e:
            print(f"Error retrieving chat history for session {session_id}: {str(e)}")
            return []


if __name__ == "__main__":
    sg = SolutionGenerator("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")

    res = sg.process_query("help me build a chatbot")

    print(res)
