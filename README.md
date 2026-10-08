# Project starter

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
