<h1 align="center">
&nbsp; Using language models <br>to detect and reduce gender bias in university forum messages<br>
  👧⚖️👦<br>
</h1>

<p align="center">
  <a href="https://github.com/GianniCatBug/spanish-gender-debias/blob/f9cd550f4b7850f8beb1f4ab3f3ecb0210b3e44b/paper-using-language-models-to-detect-and-reduce-gender-bias-in-university-forum-messages.pdf" target="_blank">[Paper]</a>
  <a href="https://github.com/GianniCatBug/spanish-gender-debias/blob/f9cd550f4b7850f8beb1f4ab3f3ecb0210b3e44b/annotators-manual.pdf" target="_blank">[Annotators Manual]</a>
  <a href="https://github.com/GianniCatBug/spanish-gender-debias" target="_blank">[Publication]</a>
</p>

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


## Citing

```
@article{salomo_SPM2025,
  author = {Salomo-López, Gianina and Alcázar, Cristóbal and Barceló, Roberto and Carvajal Reyes, Camilo and Radovic, Darinka and Tobar, Felipe},
  title = {Using language models to detect and reduce gender bias in university forum messages},
  journal = {IEEE Signal Processing Magazine},
  note = {In press},
  year = {2025}
}
```

