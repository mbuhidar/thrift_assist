# This file tells deployment platforms (like Heroku, Cloud Run, etc.) how to run your app
# Format: <process_type>: <command>
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300 --workers 1
