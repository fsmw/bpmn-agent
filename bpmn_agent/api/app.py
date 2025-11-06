"""
FastAPI application for BPMN Agent pattern matching service.

Provides REST API endpoints for pattern discovery, search, and validation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.pattern_matching_routes import router as pattern_router

# Create FastAPI app
app = FastAPI(
    title="BPMN Agent Pattern Matching API",
    description="REST API for advanced pattern matching in BPMN process extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pattern_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "BPMN Agent Pattern Matching API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
