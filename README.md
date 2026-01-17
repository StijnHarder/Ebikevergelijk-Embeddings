# eBike Multi-Shop Aggregator & Vector Matcher

This project is a high-performance data engineering pipeline designed to scrape electric bike data from multiple webshops, generate multi-modal embeddings using Meta's **PE-Core** vision-language model, and cluster identical products across different retailers using vector similarity.

## 🚀 Overview

The system solves the problem of product deduplication in the e-bike market. Since different webshops use different naming conventions and image styles, traditional SKU matching often fails. This project uses **AI-driven multi-modal embeddings** (combining Image + Title) to identify the same bike model across the web.

### Key Features
*   **Multi-Shop Scraping:** Distributed scraping logic (Scrapy) to fetch bike details, prices, and images.
*   **Multi-modal Embeddings:** Uses Meta's `PE-Core-L14-336` model to create concatenated multi-dimensional vectors capturing both visual and semantic data.
*   **Vector Search & Clustering:** Utilizes the `pgvector` extension on Supabase to store and compare high-dimensional embeddings.
*   **Automated Pipeline:** Fully configured GitHub Actions workflow for daily data synchronization and re-clustering.

---

## 🏗️ System Architecture

1.  **Scraping Layer:** Data is gathered from various shops and stored in the `scraped_bikes` table.
2.  **Embedding Layer (`backfill_vector_storage.py`):** 
    *   Identifies bikes missing embeddings via a set difference logic.
    *   Processes Product Title + Image URL through the **PE-Core Model**.
    *   Generates a concatenated `[image_emb, text_emb]` vector.
    *   Saves the resulting vector to the `vector_storage` table.
3.  **Clustering Layer (`bike_clusters.py`):** 
    *   Loads vectors into memory for high-speed processing.
    *   Uses a **Greedy Baseline approach** starting with a trusted shop (*fietsvoordeelshop.nl*) as the anchor.
    *   Calculates Cosine Similarity (threshold: **0.96**) to match bikes from other shops.
    *   Uses `upsert` logic to map bikes into the `master_bikes` table without duplicates.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.12.8
*   **Database:** Supabase (PostgreSQL + pgvector)
*   **ML Frameworks:** PyTorch 2.5.1, HuggingFace Transformers
*   **Vision Model:** Meta PE-Core (Perception Test Core Model)
*   **Automation:** GitHub Actions (CI/CD)

---

## 📊 Database Schema

### `scraped_bikes`
The raw data from web crawlers.
*   `id`: UUID (Primary Key)
*   `name`: Product title
*   `image_url`: Link to the product photo
*   `webshop`: Source shop name
*   `price`: Product price

### `vector_storage`
Stores the high-dimensional AI representations.
*   `bike_id`: Reference to `scraped_bikes` (FK)
*   `embedding`: `vector` type (Concatenated Image + Text)

### `master_bikes`
The final mapping of clustered products.
*   `cluster_id`: UUID shared by identical products across shops.
*   `bike_id`: Reference to the specific shop listing (Unique).

---

## ⚙️ Setup & Installation

### 1. Prerequisites
*   Python 3.12.x
*   Supabase project with `pgvector` enabled.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

### 3. Install Dependencies
```bash
# Install PyTorch (select version based on your local OS/GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project and model requirements
pip install -r requirements.txt
pip install -r perception_models/requirements.txt
```

---

## 🤖 Automated Workflow

The project includes a robust GitHub Action (`.github/workflows/cron_job.yml`) that runs on a schedule or manual trigger.

The workflow is optimized for:
*   **Efficiency:** Uses a CPU-only version of Torch to minimize runner disk usage and installation time.
*   **Robustness:** Filters out conflicting dependencies (like xformers) that often fail in non-GPU environments.
*   **Data Integrity:** Runs the embedding backfill first, followed by the clustering update to ensure the database is always in sync.

---

## 📝 Future Roadmap

*   Migrate clustering logic from Python memory to SQL-based similarity joins using pgvector operators (`<=>`).
*   Implement asynchronous image fetching to optimize embedding generation speed.
*   Add a weight-tuning layer to the concatenated embeddings to prioritize specific bike attributes.