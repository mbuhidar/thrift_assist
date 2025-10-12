"""
Main entry point for Render.com deployment.
Imports the FastAPI app from the modular backend package.
"""
from backend.api.main import app

__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)