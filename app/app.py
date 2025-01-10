from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from solution_generator import SolutionGenerator
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GenAI Guide API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class UserQuery(BaseModel):
    prompt: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    messages: List[Message]
    status: str = "success"


class APIKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=1)


class APIKeyResponse(BaseModel):
    message: str
    status: str = "success"


class NewChatResponse(BaseModel):
    message: str
    status: str = "success"


class ChatHistoryResponse(BaseModel):
    messages: List[dict]


# Session state with type hints
class SessionState:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.solution_generator: Optional[Any] = None
        self.messages: List[Dict[str, str]] = []


# Initialize session state
session_state = SessionState()


# Helper Functions
def get_solution_generator():
    if not session_state.api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    if session_state.solution_generator is None:
        try:
            session_state.solution_generator = SolutionGenerator(
                groq_api_key=session_state.api_key
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return session_state.solution_generator


def validate_session():
    if not session_state.api_key:
        raise HTTPException(status_code=400, detail="API key is required")


@app.post("/ask", response_model=UserResponse)
async def ask_question(query: UserQuery):
    """Handle user input and generate a response."""
    validate_session()

    user_message = {"role": "user", "content": query.prompt}
    session_state.messages.append(user_message)

    solution_generator = get_solution_generator()
    response = solution_generator.process_query(query.prompt)

    assistant_message = {"role": "assistant", "content": response}
    session_state.messages.append(assistant_message)

    return {
        "messages": [Message(**msg) for msg in session_state.messages],
        "status": "success",
    }


@app.post("/set-api-key", response_model=APIKeyResponse)
async def set_api_key(request: APIKeyRequest):
    """Set the API key for the session and initialize the SolutionGenerator."""
    session_state.api_key = request.api_key
    try:
        session_state.solution_generator = SolutionGenerator(
            groq_api_key=session_state.api_key
        )
    except Exception as e:
        session_state.api_key = None  # Reset API key if initialization fails
        raise HTTPException(
            status_code=400, detail=f"Failed to initialize SolutionGenerator: {str(e)}"
        )

    return {
        "message": f"API key has been set and SolutionGenerator initialized successfully with sessionID: {session_state.solution_generator.session_id}",
        "status": "success",
    }


@app.post("/new-chat", response_model=NewChatResponse)
async def new_chat():
    """Reset the chat session except for the API key and reinitialize the SolutionGenerator."""
    if not session_state.api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    # Reset session state except for the API key
    session_state.solution_generator = SolutionGenerator(
        groq_api_key=session_state.api_key
    )
    session_state.messages = []

    return {
        "message": f"New chat session started successfully with sessionID: {session_state.solution_generator.session_id}",
        "status": "success",
    }


@app.get("/retrieve_chat_history", response_model=ChatHistoryResponse)
def retrieve_chat_history(session_id: str):
    """
    Endpoint to retrieve chat history for a given session ID.
    """
    try:
        messages = session_state.solution_generator.retrieve_chat_history(session_id)
        if not messages:
            raise HTTPException(
                status_code=404,
                detail=f"No chat history found for session_id: {session_id}",
            )
        return {"session id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
