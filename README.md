# Teaching Agent with Mistral 7B Instruct and RAG

This project implements a teaching agent that uses Mistral 7B Instruct model with Retrieval-Augmented Generation (RAG) to provide intelligent responses based on course materials.

## Features

- Upload course materials (text documents)
- Process documents using RAG
- Answer questions based on uploaded course content
- FastAPI-based REST API
- Vector store for efficient retrieval

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /upload_course_material` - Upload course material documents
- `POST /ask_question` - Ask questions about the course material

## Usage Example

1. Upload a course material:
```bash
curl -X POST http://localhost:8000/upload_course_material \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/course_material.txt"
```

2. Ask a question:
```bash
curl -X POST http://localhost:8000/ask_question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main concept discussed in the course?"}'
```

## Project Structure

- `app.py` - Main FastAPI application
- `requirements.txt` - Project dependencies
- `.env` - Configuration file
- `course_materials/` - Directory for storing uploaded course materials
- `chroma_db/` - Directory for vector store data

## Requirements

- Python 3.8+
- CUDA (for GPU acceleration)
- Sufficient RAM (at least 16GB recommended)
- GPU with at least 16GB VRAM (recommended for Mistral 7B)
