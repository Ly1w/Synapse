import os

SERPER_API_KEY = os.environ.get(
    "SERPER_API_KEY",
    "55b1a99aa3e19ff3b5cbb86ad58234d14ffad37e",
)

SERPER_BASE_URL = "https://google.serper.dev"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Directory for persisting notes, sources, and reports
STORAGE_DIR = os.path.expanduser("~/.deep_research")
