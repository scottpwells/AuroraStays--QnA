FROM python:3.13.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Keep the build context limited to application artifacts; credentials are runtime secrets.
COPY app.py rag_utils.py policy_retrieval.py deployment_manifest.json ./
COPY faiss_index/ ./faiss_index/

ENV HF_HOME=/app/huggingface
RUN python -c "import json; from langchain_huggingface import HuggingFaceEmbeddings; m=json.load(open('deployment_manifest.json')); HuggingFaceEmbeddings(model_name=m['embedding_model'], model_kwargs={'device':'cpu','revision':m['embedding_revision']}, encode_kwargs={'normalize_embeddings':True})"

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:create_app()"]
