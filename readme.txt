Start project:
1. Start Chroma DB server:
   - `docker run -p 8000:8000 ghcr.io/chroma-core/chroma:latest`

2. python ./scripts/test_search.py
3. check chroma db: curl http://localhost:8000/api/v2/heartbeat
