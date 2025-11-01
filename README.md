# local-rag-app-example
Local rag app example

## Setup

```bash
uv init --app --python 3.12
```

```bash
uv add openai
```

```bash
uv add --dev ruff black
```

### Install dependencies

```bash
uv sync
```

## Run

```bash
uv run ruff check .
```

```bash
uv run black .
```

```bash
uv run python main.py
```
