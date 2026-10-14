# locsmith

Local, private pre-translation of game strings with an open-weight model.

Indie teams can't send an unannounced game's script to a cloud API (NDAs, leaks) and can't afford
an agency pass on every build. `locsmith` runs a first translation pass on your own machine,
then **validates every string** so the output is safe to drop into the build:

- placeholders and tags survive untouched: `{0}`, `{name}`, `%s`, `%1$s`, `<color=#f00>`, `[b]`, `\n`
- UI length limits (`max_len` per string)
- glossary terms (character names, currencies) used consistently; inflected languages supported via stems, `strict` for exact names
- broken strings are sent back to the model with the exact problem, up to `--retries` times; the rest are flagged `review`
- reruns only translate new or changed strings

Works with any OpenAI-compatible endpoint: Ollama, llama.cpp `llama-server`, vLLM, LM Studio. Standard library only.

Gemma 4 and other reasoning models "think" before answering, which is wasted effort for structured translation. locsmith sends `reasoning_effort: none` by default, cutting a batch from thousands of tokens to a few hundred (~10x faster); raise it with `--reasoning low|medium|high` if you want the model to deliberate.

## Quick start

```bash
ollama pull gemma4:12b          # or any other Gemma tag you have
python3 locsmith.py samples/strings.csv --to ru -g samples/glossary.ru.csv
# -> samples/strings.ru.csv  (key,source,target,status,issues)
```

llama.cpp instead of Ollama:

```bash
llama-server -m gemma.gguf --port 8080
python3 locsmith.py samples/strings.csv --to ru --base-url http://localhost:8080/v1 --model local
```

## Input

CSV with `key,source` and optional `context` (meaning/tone hint) and `max_len`, or a flat JSON `{key: text}`.
Glossary CSV: `source,target[,strict]`.

Exit code is `2` if any string needs human review, so it can gate a CI step.

## Tests

```bash
python3 -m unittest discover -s tests
```

## License

MIT
