# Scripture Detector

> **AI-powered detection and analysis of biblical quotations, paraphrases, and allusions in historical texts.**

![Sources page](static/screenshots/sources.png)

Developed by **Dr. William J.B. Mattingly**, Cultural Heritage Data Scientist at Yale University.

Built for the international workshop **[Ruse of Reuse: Detecting Text-similarity with AI in Historical Sources](https://www.oeaw.ac.at/en/imafo/events/event-details/ruse-of-reuse)** — held **March 5–6, 2026** at the Austrian Academy of Sciences, Vienna.

---

## Overview

Scripture Detector is a local web application that uses Google Gemini to automatically find, classify, and annotate every biblical reference in any text you provide — from full verbatim quotations to subtle allusions. It renders an interactive, color-coded view of the source text with side-by-side Bible verse lookup, distribution charts, and a manual annotation editor.

**The only external dependency is a free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey). No cloud account, credit card, or additional infrastructure is required.

---

## Screenshots

### Sources Page
Manage all your text sources and see high-level detection statistics at a glance.

![Sources page](static/screenshots/sources.png)

### Text Viewer
Color-coded annotations appear directly in the text. Click any highlighted passage to see the matched Bible verse(s) in the panel on the right.

![Viewer page](static/screenshots/viewer.png)

### Analytics Dashboard
Explore scripture distribution across all sources — broken down by Bible book, testament, and quote type.

![Dashboard](static/screenshots/dashboard.png)

### Settings
Configure your Gemini API key or Google Vertex AI credentials, and select which Gemini model to use.

![Settings page](static/screenshots/settings.png)

### About Page
Full documentation of the application, its features, and how it works.

![About page](static/screenshots/about.png)

---

## Features

- **Full-document AI analysis** — send any text to Gemini and receive a structured list of every scripture reference with verse citations and classification types.
- **Four classification types**: Full, Partial, Paraphrase, and Allusion.
- **Color-coded in-text highlighting** — each type is rendered in a distinct color directly within the source text.
- **Click-to-explore interactivity** — click a highlighted passage to highlight the corresponding annotation card, and vice versa.
- **Selection-based re-analysis** — highlight any portion of the text to re-run AI detection on just that selection.
- **Manual annotation editor** with an integrated Bible verse picker (book → chapter → verse).
- **Distribution charts** per source and across all sources (by Bible book, testament, and type).
- **Global analytics dashboard** aggregating data across all sources.
- **Model switching** — choose between available Gemini model versions in the UI.
- **Runs entirely locally** — your texts are never transmitted anywhere except for the Gemini API call itself.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Install & Run

```bash
git clone <repo-url>
cd scripture-detector

# Using uv (recommended)
uv run python app.py

# Or using pip
pip install -e .
python app.py
```

The app starts at **http://127.0.0.1:5001**.

### 3. Configure API Key

1. Open [http://127.0.0.1:5001/settings](http://127.0.0.1:5001/settings)
2. Select **Gemini API** as the provider
3. Paste your API key from [Google AI Studio](https://aistudio.google.com/apikey)
4. Click **Save Settings**

### 4. Analyze a Text

1. Go to the **Sources** page and click **Add Source**
2. Give your source a name and paste in the text to analyze
3. Click **View** to open the text viewer
4. Click **Process with AI** — Gemini will detect all scripture references within seconds
5. Click any highlighted passage to explore the matched Bible verses

---

## Quote Classification

| Type | Description |
|---|---|
| **Full** | A complete or near-complete verse quoted verbatim |
| **Partial** | A recognizable portion of a verse with minor variation or truncation |
| **Paraphrase** | Biblical content restated in different words, preserving the meaning |
| **Allusion** | A brief phrase, thematic echo, or indirect reference to a specific verse |

---

## Project Structure

```
scripture-detector/
├── app.py              # Flask application and API routes
├── database.py         # SQLite database layer
├── main.py             # CLI batch evaluation script
├── data/
│   ├── bible.tsv       # Full Bible verse database (35,000+ verses)
│   └── book_mapping.tsv
├── templates/
│   ├── sources.html    # Sources listing page
│   ├── viewer.html     # Annotated text viewer
│   ├── dashboard.html  # Global analytics dashboard
│   ├── settings.html   # API configuration
│   └── about.html      # About / documentation page
└── static/
    ├── style.css        # Yale color palette stylesheet
    ├── logo.svg         # Application logo
    └── favicon.svg      # Browser tab icon
```

---

## API Providers

### Gemini API (Free Tier)
The simplest option. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey) — no billing required. Select **Gemini API** in Settings and paste your key.

### Google Vertex AI
For enterprise use or higher rate limits. Requires a Google Cloud project with Vertex AI enabled. Select **Vertex AI** in Settings and enter your project ID and location.

---

## About

**Developer:** Dr. William J.B. Mattingly  
**Affiliation:** Yale University, Cultural Heritage Data Scientist  
**Workshop:** [Ruse of Reuse: Detecting Text-similarity with AI in Historical Sources](https://www.oeaw.ac.at/en/imafo/events/event-details/ruse-of-reuse)  
**Workshop Dates:** March 5–6, 2026  
**Workshop Venue:** Austrian Academy of Sciences, PSK Georg-Coch-Platz 2, 1010 Vienna  
**Organisers:** Digital Lab, Institute for Medieval Research, Austrian Academy of Sciences & [SOLEMNE](https://canones.org/), Radboud University  

---

## License

MIT
