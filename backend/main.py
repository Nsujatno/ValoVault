from fastapi import FastAPI
from db import supabase
from fastapi.middleware.cors import CORSMiddleware

from routes import plays
app = FastAPI(
    title="Valorant Playbook API",
    description="RAG-based playbook query system for Valorant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plays.router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"Hello": "World"}