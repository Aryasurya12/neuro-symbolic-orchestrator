from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as rest_router
from .websocket import router as ws_router

app = FastAPI(
    title="Neuro-Symbolic Cloud FinOps Gateway",
    description="FastAPI + WebSocket gateway for the Symbolic Optimization Engine.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router)
app.include_router(ws_router)
