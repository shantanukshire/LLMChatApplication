# LLM Chat application

## How to run

1. **Prerequisites:** Python 3.10+.

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   # Windows (PowerShell or CMD):
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set required environment variables:**
   - `OPENAI_API_KEY` — your OpenAI API key (used for the LLM).
   - `ELYOS_AI_API_KEY` — API key for the weather and research tools.

   Example (Windows PowerShell):

   ```powershell
   $env:OPENAI_API_KEY = "your-openai-key"
   $env:ELYOS_AI_API_KEY = "your-elyos-key"
   ```

   Example (macOS/Linux):

   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ELYOS_AI_API_KEY="your-elyos-key"
   ```

5. **Run the application:**

   ```bash
   python main.py
   ```

6. **In the chat:** Type your message and press Enter. Type `quit`, `exit`, or `q` to exit. Use **Ctrl+C** to cancel a long-running tool (e.g. research).

## Running tests

From the project root (with dependencies installed):

```bash
# Run all tests
python -m unittest discover -s tests -v
```

---

## Features

1. Accepts text input from the user
2. Sends input to an LLM (OpenAI, Anthropic, or similar)
3. **Streams** the response back to the terminal in real-time
4. Supports **tool calling** with two APIs:

- A weather API (usually fast, ~200ms)
- A "research" API (slow, 3-8 seconds)

5. Handles **pending states** — show the user something is happening during slow tool calls
6. Supports **cancellation** — user can interrupt a long-running operation (Ctrl+C or similar)
7. **Handles the APIs gracefully** — these are real-world APIs with real-world quirks

### Example Interactions

```
You: What's the weather in London?
Assistant: [calls get_weather]
London: **5.3 °C**, **partly cloudy**, **93% humidity**.

You: Research quantum computing
Assistant: [calls research_topic]
Researching quantum computing... (Ctrl+C to cancel)
[Ctrl+C pressed]
Research tool cancelled.
Operation cancelled.


You: Research medical science
Assistant: [calls research_topic]
Researching medical science... (Ctrl+C to cancel)
[Ctrl+C pressed]
Research tool cancelled.
Operation cancelled.


You: exit

Exiting...
```
