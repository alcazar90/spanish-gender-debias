# spanish-gender-debias

# Install packages

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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