<h1 align="center">
  &nbsp; Mitigating gender bias in STEM education<br>using an LLM-based corrector for student communication<br>
  👧⚖️👦<br>
</h1>


# Install packages

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Ensure you have set your API keys correctly in your `.env` file, you can find an exmaple env file in the repo: `.example-env`.


# Web

1. Go to folder
```bash
cd web/
```

2. Create `.env` file (example available in `env_example`)

3. Start API with
```bash
uvicorn api:app --reload
```

4. Open `index.html` in browser
