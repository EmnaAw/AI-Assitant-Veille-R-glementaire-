# Preventis Veille

Preventis Veille is a regulatory monitoring and compliance application. It combines a React dashboard, a Spring Boot API, Python AI services, a legal recommendation engine, and an Ollama model runtime.

The application is designed for teams that need to manage regulatory texts, articles, legal terms, clients, conformity status, action plans, notifications, contacts, and AI-assisted regulatory questions.

## Project Structure

```text
D:\PFE
|-- Application-Veille-Front
|   React + Vite dashboard used by admins, clients, and super clients.
|
|-- Application-Veille-Back
|   Spring Boot backend: authentication, business APIs, database access,
|   mail, notifications, JORT workflow trigger, and AI service integration.
|
|-- AI-Assitant-Veille-R-glementaire-
|   |-- AI-Engine
|   |   FastAPI service for RAG and AI orchestration.
|   |
|   |-- Recommendation-system
|       Hybrid legal recommendation engine using deterministic lookup,
|       retrieval, reranking, and controlled generation.
|
|

```

## Main Runtime Pieces

- Frontend: Vite dev server on `http://localhost:3000`.
- Backend: Spring Boot API on `http://localhost:9090`.
- AI Engine: FastAPI on `http://localhost:8000`.
- Ollama: local model server on `http://localhost:11434`.
- Database: MySQL database configured through backend environment variables.

The backend calls the AI Engine through `AI_PYTHON_BASE_URL`. The AI Engine calls Ollama through `OLLAMA_BASE_URL`.

## Prerequisites

For running from source:

- Windows PowerShell
- Java 17
- Maven, or the included `Application-Veille-Back\mvnw.cmd`
- Node.js 20+
- Python 3.11+
- Ollama
- MySQL access
- Optional: `cloudflared` for Cloudflare tunnel hosting



## Environment File

Create a root `.env` file at:

```text
D:\PFE\.env
```

Do not commit this file. It contains passwords, API keys, database access, and model settings.

Use this as a safe template and replace every placeholder:

```env
# Ports
FRONTEND_PORT=3000
BACKEND_PORT=9090
AI_ENGINE_PORT=8000
OLLAMA_PORT=11434

# Backend to AI engine
AI_PYTHON_BASE_URL=http://127.0.0.1:8000
AI_PYTHON_CONNECT_TIMEOUT_MS=10000
AI_PYTHON_READ_TIMEOUT_MS=180000
AI_ACCESS_DEFAULT_ENABLED=true

# Ollama / generation
GENERATION_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=vigogne-llama-3:latest
RAG_LLM_MODEL=vigogne-llama-3:latest
RECOMMENDATION_OLLAMA_MODEL=vigogne-llama-3:latest
OLLAMA_API_KEY=
OLLAMA_TIMEOUT=120
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=384
OLLAMA_NUM_GPU=0

# Embeddings
RAG_EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_LOCAL_FILES_ONLY=false
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0

# JORT workflow
JORT_WORKFLOW_ENABLED=false
JORT_WORKFLOW_PYTHON=python
JORT_WORKFLOW_BASE_DIR=./jort-workflow
JORT_WORKFLOW_SCRIPT_PATH=monitor_jort.py
JORT_NOTIFICATION_API_KEY=replace-if-jort-is-enabled

```
## Manual Source Commands

Use these when you want to debug one service at a time.

### 1. Python AI Environment

```powershell
cd D:\PFE\AI-Assitant-Veille-R-glementaire-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\AI-Engine\requirements
pip install -r .\Recommendation-system\requirements.txt
```

### 2. Ollama

```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"
ollama serve
```

In another terminal, make sure the required models exist:

```powershell
ollama list
```

If your models are custom, import or create them before starting the full app.

### 3. AI Engine

```powershell
cd D:\PFE\AI-Assitant-Veille-R-glementaire-\AI-Engine
$env:AI_PRELOAD_SERVICES="0"
python -m uvicorn service_api:app --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Useful endpoints:

- `POST /ai/respond`
- `POST /ai/rag/respond`
- `POST /ai/recommendation/respond`
- `GET /ai/health`

### 4. Backend

```powershell
cd D:\PFE\Application-Veille-Back
$env:SPRING_PROFILES_ACTIVE="dev"
$env:AI_PYTHON_BASE_URL="http://127.0.0.1:8000"
$env:JORT_WORKFLOW_ENABLED="false"
.\mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=dev
```

Health check:

```text
http://127.0.0.1:9090/actuator/health
```

### 5. Frontend

```powershell
cd D:\PFE\Application-Veille-Front
npm install
npm run dev:local
```

Open:

```text
http://localhost:3000
```

### 6. Optional Recommendation Streamlit UI

```powershell
cd D:\PFE\AI-Assitant-Veille-R-glementaire-
.\.venv\Scripts\Activate.ps1
streamlit run .\Recommendation-system\app\app.py
```
## Hosting Alternative 1: PC Hosting With Cloudflare Tunnel

This option is best for demos, temporary access, and important meetings where the app needs to be reachable from outside the local network without deploying to a permanent server.

Runtime path:

```text
External user
-> Cloudflare Tunnel
-> host PC
-> frontend localhost:3000
-> backend localhost:9090
-> AI engine localhost:8000
-> Ollama localhost:11434
```

Rules for this option:

- Keep the host PC awake.
- Keep internet stable.
- Keep Ollama on localhost. Do not expose raw Ollama directly to the internet.
- Prefer a named Cloudflare tunnel with your own domain for serious meetings.
- Use quick tunnels only when you need a fast temporary URL.

### Recommended: Named Cloudflare Tunnel

Create two public hostnames in Cloudflare:

```text
app.your-domain.com -> http://localhost:3000
api.your-domain.com -> http://localhost:9090
```

Install `cloudflared` as a service using the command Cloudflare gives you:

```powershell
cloudflared.exe service install <TUNNEL_TOKEN>
```

Start the local stack using the public URLs:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
D:\PFE\Start-Preventis-PC.ps1 `
  -PublicApiBaseUrl "https://api.your-domain.com" `
  -PublicFrontendUrl "https://app.your-domain.com"
```

Give meeting users:

```text
https://app.your-domain.com
```

If CORS needs to be tightened for this domain, set:

```powershell
$env:APP_CORS_ALLOWED_ORIGIN_PATTERNS="https://app.your-domain.com,https://*.trycloudflare.com,http://localhost:3000,http://127.0.0.1:3000"
```

### Fast Fallback: Cloudflare Quick Tunnel

Quick tunnel URLs are random and temporary. Use them only for short sessions.

Start the local app in single-public-URL mode:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
D:\PFE\Start-Preventis-PC.ps1 `
  -PublicApiBaseUrl "/." `
  -PublicFrontendUrl "http://localhost:3000"
```

Then start the quick tunnel:

```powershell
D:\PFE\Start-Cloudflare-QuickTunnel.ps1
```

The script prints a URL like:

```text
https://something-random.trycloudflare.com
```

Give that URL to meeting users. Keep the PowerShell windows, local services, and PC running until the meeting ends.

To reduce sleep risk on the host PC, run PowerShell as Administrator:

```powershell
powercfg /hibernate off
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

## Hosting Alternative 2: Company Server Hosting

This option is best for stable internal or production use. The company server should run the services continuously, expose only HTTPS to users, and keep internal service ports private.

Recommended high-level architecture:

```text
Users
-> company domain over HTTPS
-> reverse proxy / load balancer
-> frontend static build
-> Spring Boot backend
-> Python AI Engine
-> Ollama
-> MySQL
```

Major steps:

1. Prepare the server.
   - Install Docker and Docker Compose, or install Java 17, Node.js 20, Python 3.11, Ollama, and MySQL directly.
   - Create a dedicated application user.
   - Open only `80` and `443` publicly.
   - Keep backend `9090`, AI `8000`, Ollama `11434`, and MySQL private.

2. Configure domain and TLS.
   - Point `veille.your-company.com` to the server.
   - Use Nginx, Apache, Caddy, or Cloudflare as the HTTPS reverse proxy.
   - Add TLS certificates with Let's Encrypt or the company certificate manager.

3. Prepare environment variables.
   - Create a server `.env` with production database, JWT, mail, AI, CORS, and model settings.
   - Set `SPRING_PROFILES_ACTIVE=prod` for direct Spring Boot production mode, or `SPRING_PROFILES_ACTIVE=docker` when using the Docker profile.
   - Set frontend build values:

```env
VITE_NODE_ENV=production
VITE_API_BASE_URL=https://api-or-backend-domain.example.com
VITE_FRONTEND_URL=https://veille.your-company.com
```

4. Prepare the database.
   - Create or migrate the MySQL database.
   - Apply the SQL migration files in `Application-Veille-Back` if the target database is empty or missing required columns.
   - Create a least-privilege database user for the app.
   - Schedule backups.

5. Deploy the application.
   - Docker path: build/push frontend, backend, and AI Engine images, then run a production Docker Compose file on the server.
   - Direct path: build the frontend with `npm run build`, serve `dist` through Nginx, build the backend jar with Maven, and run AI Engine with Uvicorn.

6. Configure AI models.
   - Install Ollama on the server or run it as a private container.
   - Import or pull the required models.
   - Keep `OLLAMA_BASE_URL` private, usually `http://127.0.0.1:11434` or `http://ollama:11434` inside Docker.

7. Run services permanently.
   - Use Docker restart policies, systemd services, Windows services, or a supervisor.
   - Store logs centrally.
   - Add health checks for frontend, backend `/actuator/health`, and AI `/health`.

8. Secure and operate.
   - Rotate `JWT_SECRET_KEY`, database passwords, mail passwords, and API keys.
   - Restrict CORS to the final company domain.
   - Do not expose Ollama publicly.
   - Set up backups, monitoring, and a rollback plan.

## Build And Test Commands

Frontend:

```powershell
cd D:\PFE\Application-Veille-Front
npm run build
npm run test:run
```

Backend:

```powershell
cd D:\PFE\Application-Veille-Back
.\mvnw.cmd test
.\mvnw.cmd -DskipTests package
```

Recommendation system:

```powershell
cd D:\PFE\AI-Assitant-Veille-R-glementaire-\Recommendation-system
python main.py health
python main.py index
python main.py query "absence d'autorisation administrative"
pytest
```

AI Engine:

```powershell
cd D:\PFE\AI-Assitant-Veille-R-glementaire-\AI-Engine
python -m uvicorn service_api:app --host 127.0.0.1 --port 8000
```

## Operational Notes

- For meetings, use the PC + Cloudflare named tunnel option when you need quick external access.
- For real company usage, prefer server hosting with HTTPS, private internal ports, backups, monitoring, and stable domains.
- Never expose the Ollama port directly to the public internet.
- Keep `.env` files out of Git.
- After changing frontend public URLs, restart the frontend because Vite environment values are injected at startup/build time.
- After changing backend environment variables, restart Spring Boot.
- After changing AI model or embedding settings, restart the AI Engine.
