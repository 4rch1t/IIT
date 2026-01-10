# kds_hackathon

Project scaffold for claim extraction and scoring from novels.

Structure:

kds_hackathon/
├── data/
│   ├── novels/
│   └── backstories/
├── ingestion/
│   └── ingest_novels.py
├── claims/
│   └── extract_claims.py
├── retrieval/
│   └── retrieve_evidence.py
├── scoring/
│   └── local_claim_scoring.py
├── aggregation/
│   └── global_decision.py
├── pipeline/
│   └── run_pipeline.py
├── results/
│   └── results.csv
├── report/
│   └── report.md
├── requirements.txt
└── README.md

Run the pipeline:

```powershell
python pipeline/run_pipeline.py
```
