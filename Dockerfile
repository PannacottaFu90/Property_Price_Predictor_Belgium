# 1. Usa un'immagine Python leggera (Slim) per ridurre lo spazio
FROM python:3.10-slim

# 2. Imposta la cartella di lavoro dentro il container
WORKDIR /app

# 3. Installa le dipendenze di sistema necessarie (opzionale ma consigliato)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/apt/lists/*

# 4. Copia il file dei requisiti e installa le librerie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia tutto il resto del progetto (cartelle src, model, app.py, ecc.)
COPY . .

# 6. Esponi la porta su cui girerà FastAPI (Render usa la 10000 di default o $PORT)
EXPOSE 8000

# 7. Comando per avviare l'API
# Usiamo 0.0.0.0 per accettare connessioni esterne
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
