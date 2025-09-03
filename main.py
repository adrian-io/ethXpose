from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.tools.classify import classify_wallet
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI(
    title="ETHXpose - Ethereum Wallet Fraud Detection API",
    version="1.0",
    docs_url="/api/docs",           # Swagger UI
    openapi_url="/api/openapi.json" # OpenAPI schema
)

# Allow frontend origins (during development allow localhost:3000)
origins = [
    "https://ethxpose.vercel.app",
    "https://ethxpose-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all origins (not recommended in prod)
    allow_credentials=True,
    allow_methods=["*"],     # allow all HTTP methods
    allow_headers=["*"],     # allow all headers
)

# Define data models
class Node(BaseModel):
    id: str
    label: str

class Edge(BaseModel):
    source: str
    target: str
    value: float
    timestamp: str  # Expected as ISO 8601 formatted string

class GraphData(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

class ClassificationResponse(BaseModel):
    fraud_probability: float
    graph: GraphData

class ClassifyRequest(BaseModel):
    wallet_address: str
    model_name: str = "first_Feather-G_GB.joblib"  # Default value for model_name

# Define classify_wallet endpoint
@app.post("/api/py/classify", response_model=ClassificationResponse)
def classify_wallet_endpoint(request: ClassifyRequest):
    """
    Classify a wallet based on its transaction graph and return fraud probability and graph data.

    Args:
        request (ClassifyRequest): The request body containing wallet_address and model_name.

    Returns:
        dict: A response with the fraud probability and graph data.
    """
    try:
        # Call the classify_wallet function
        fraud_probability, graph_data = classify_wallet(request.wallet_address, request.model_name)

        if fraud_probability is None or graph_data is None:
            raise ValueError("Classification failed. Please check the wallet address or model.")

        # Structure and return the response
        response = {
            "fraud_probability": fraud_probability,
            "graph": graph_data
        }
        print(response)
        return response
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
