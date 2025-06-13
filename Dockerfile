# Base image from official Chroma server
FROM ghcr.io/chroma-core/chroma:latest

# Optional: set environment variables
ENV CHROMA_SERVER_HOST=0.0.0.0
ENV CHROMA_SERVER_PORT=8000

# Default command to run Chroma server
CMD ["chromadb", "run"]
