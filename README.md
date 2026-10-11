# FoodAdvisor

FoodAdvisor retrieves complete recipe chunks from cookbook resources, extracts evidence-backed ingredients with a first LLM call, and identifies allergens in a separate second call. The English-language page presents the allergen result and warnings, with a Backend log showing the chunks and JSON payloads exchanged at each stage.

The home page includes an action to reload recipe resources and rebuild the RAG index.

Recipe names entered by users and all cookbook text are treated as untrusted data, never as instructions. The analysis stages validate JSON responses and report errors instead of silently accepting malformed output. See the [requirements](./requirements.md), [agent specification](./agents.md), and [project skills](./.github/skills/README.md) for details.

## Start the application

The application needs a Python API service and a Vite web service. The recipe index is local; the language-model calls use the configured remote API. No database or separate RAG service needs to be started.

Install the prerequisites once from the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r backend/requirements.txt
cd frontend
npm ci
cd ..
```

Copy and configure `.env` as described below, then start both services together from the project root:

```bash
./start.sh
```

Open <http://localhost:5173> in your browser. The API listens on port `8000`; keep the terminal open while using the app and press `Ctrl+C` to stop both services. If using a development container, forward ports `5173` and `8000`.

The first recipe search may take longer while the local RAG index and embedding model are prepared. Later searches reuse the index; the home page also provides an action to rebuild it after changing recipe resources. To use a different Python executable, set `PYTHON_BIN` when launching the script.

## Add your API key locally

You will receive your group's API key by e-mail. They go in a local `.env` file, which Git ignores.

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and paste the key from the e-mail:

   ```
   LLM_API_KEY=...
   ```

   The proxy URL and model name are already built in; you do not need to set them.

3. Save the file. Do not rename it, and do not move the key anywhere else (code, notes, chat, issues).

4. Check that Git is not tracking it. This command must **not** list `.env`:

   ```bash
   git status
   ```

Never commit `.env` or paste the key into any file that is committed. If you think the key has leaked, e-mail the instructor straight away.
