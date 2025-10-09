"""
Run the ThriftAssist API server.
"""

import uvicorn
import os
from backend.core.config import settings

if __name__ == "__main__":
    # Get port from environment or settings
    port = int(os.getenv("PORT", settings.PORT))
    
    print(f"🚀 Starting ThriftAssist API on {settings.HOST}:{port}")
    print(f"📚 API Documentation: http://{settings.HOST}:{port}/docs")
    print(f"🌐 Web Interface: http://{settings.HOST}:{port}/")
    
    uvicorn.run(
        "backend.api.main:app",
        host=settings.HOST,
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
