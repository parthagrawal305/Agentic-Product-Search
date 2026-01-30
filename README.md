<h1 align="center">🛍️ Semantic Shopping AI Agent</h1>

<p align="center">
  A stateful, multi-agent E-commerce Shopping Assistant built on <b>LangGraph</b>, <b>FastAPI</b>, <b>Qdrant</b>, and <b>Supabase</b>.
</p>

## The "Why"
Traditional E-commerce relies on users manually clicking checkboxes (e.g., ✅ "Black", ✅ "Under $100") and scrolling through paginated SQL results. This project introduces a conversational **Semantic Routing Engine**. Users state their intent naturally ("I need a cheap, dark winter jacket for a ski trip"), and the intelligent LangGraph state machine handles the hybrid vector/relational querying autonomously.

## Architecture

1. **State Orchestrator (LangGraph):** Manages conversational flow, memory, and cart state across multiple agent nodes.
2. **The LLM Brain (Groq):** Llama-3.3-70B running at 0.0 temperature processes structured Pydantic outputs to extract exact user constraints (e.g. `max_price_inr: 100`).
3. **The Vector Engine (Qdrant + FastEmbed):** Embedded high-dimensional product matrices queried for semantic similarity.
4. **Relational Constraints (Supabase):** PostgreSQL handles the hard boundaries (e.g. strict price limits) wrapped inside Qdrant Hybrid filters.

## Local Setup (Lightning Fast with `uv`)

This project uses `uv`, an extremely fast Python package and project manager written in Rust.

1. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone and Setup:**
   ```bash
   git clone https://github.com/yourusername/Shopping-AI-Agent.git
   cd Shopping-AI-Agent
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
3. **Configure Environment:**
   Create a `.env` file from the `.env.example` and add your Groq/Supabase keys.
   
4. **Seed the Databases (One-Time):**
   ```bash
   uv run scripts/data_seeder.py
   ```
   *This automatically generates realistic e-commerce data and syncs your local Qdrant vectors with your Supabase tables.*

5. **Run the Front-End Chat:**
   ```bash
   uv run streamlit run app.py
   ```

## Live Deployment
[Live Streamlit Application](https://streamlit.io/) <!-- Replace with actual deployed URL -->
