# SmartGen AI — Resource-Aware Image Generation
**Shakti Hack 26 | PS1**

SmartGen estimates resource usage before generating an image. The user must explicitly approve the estimate; choosing “No, edit prompt” does not generate an image.

## Features
- Professional login page and session-protected app routes.
- Purpose-aware settings for coloring pages, flashcards and posters.
- Before generation, show estimated energy (Wh), time, compute, electricity cost (₹), CO₂ and a phone-charge equivalent.
- Explicit **OK, generate image** approval. **No, edit prompt** returns to the editable prompt without generating an image.
- Coloring pages use black outlines on a white background; real-model prompts request black-and-white line art.
- Resource receipt comparing smart settings with maximum-quality poster settings.
- District dashboard simulation, reusable-image library and savings statistics.

## Run locally in VS Code
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.

### Local demo login
- Email: `admin@smartgen.ai`
- Password: `SmartGen@2026`

Override before sharing/deploying: set `SMARTGEN_USERNAME`, `SMARTGEN_PASSWORD`, and a long random `FLASK_SECRET_KEY`. Optional `ELECTRICITY_RUPEES_PER_KWH` defaults to 8. Optional `HF_TOKEN`/`HF_API_TOKEN` and `AI_MODEL` configure Hugging Face image inference.

## Test
```bash
python test_app.py
```

## GitHub push
Never commit real API tokens or production secrets. `.env`, virtual environments and Python cache files are ignored.
```bash
git init
git add .
git commit -m "Add SmartGen resource-aware image generation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Files
- `app.py` — Flask routes, login, image generation, estimates and dashboard.
- `templates/login.html` — responsive login page.
- `templates/index.html` — creation flow, approval and dashboard.
- `test_app.py` — automated functional checks.

## Accuracy notes
The built-in renderer is a local mock unless inference is configured. Energy, time, CO₂ and equivalent values are model estimates, not live meter readings. Electricity cost uses a configurable tariff and is not the AI provider's API charge. The district dashboard is simulated, not live district data.
