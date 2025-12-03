# NBA Passing Network Analysis (2015–2021)

This project analyzes NBA team performance using *passing-network structures* derived from play-by-play data.  
It answers three research questions:

1. **RQ1:** Do cohesive passing networks (density, clustering, reciprocity) associate with team success?  
2. **RQ2:** How do star-player departures affect network structure over time?  
3. **RQ3:** Which network features best predict wins?

The project includes:
- A full data-processing pipeline (`master.py`)
- Modular visualization pages (`viz_rq1.py`, `viz_rq2.py`, `viz_rq3.py`)
- A complete interactive notebook (`analysis.ipynb`)
- Cleaned, minimal code after removing unused modules

---

# 1. Installation

Clone the repo:
```
git clone https://github.com/<your-repo>/NBA-Project.git
cd NBA-Project
```

Create a Python environment:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# 2. Download the Data (REQUIRED)

The dataset **is NOT included in this repository** because of Kaggle licensing.

Download manually from Kaggle:

▶ **Dataset:**  
https://www.kaggle.com/datasets/schmadam97/nba-play-by-play-data

After downloading:

1. Unzip the file  
2. Place the resulting **NBA-Data** folder into the project root directory:

```
NBA-Project/
    NBA-Data/
        2015_playbyplay.csv
        2016_playbyplay.csv
        ...
```

Correct folder structure:
```
NBA-Project/
    master.py
    analysis.ipynb
    NBA-Data/
    viz_rq1.py
    viz_rq2.py
    viz_rq3.py
    viz_utils.py
    team_games.py
    ...
```

---

# 3. Running the Full Analysis Pipeline (Python Script)

With your virtual environment activated, run:
```
python master.py
```

This performs:

- Data loading  
- Player & team event reconstruction  
- Star-player detection  
- Passing network generation  
- Network metric computation  
- Event-study panel building  
- RQ1, RQ2, RQ3 visualization generation

Plots will open automatically.

---

# 4. Running the Analysis Notebook

Launch Jupyter:
```
jupyter notebook
```

Open:
```
analysis.ipynb
```

The notebook allows step-by-step exploration of:

- Passing network construction  
- Star-player departure impacts  
- Visualization of all research questions  
- Regression-based prediction of wins  

---

# 5. Project File Overview

```
analysis.ipynb         → Interactive Jupyter analysis
master.py              → Main processing/visualization pipeline

pbp_loader.py          → Loads raw Kaggle play-by-play CSV files
player_events.py       → Creates long-form event structure
stars.py               → Flags star players based on usage percentiles
team_games.py          → Reconstructs game timelines for each team
appearances_and_departures.py → Detects absences & star departures
outcomes.py            → Win/loss & point differential computations

network_metrics.py     → Creates passing edges and computes network metrics
event_study.py         → Builds event-study windows around departures

viz_utils.py           → Shared plotting utilities
viz_rq1.py             → RQ1 visualizations
viz_rq2.py             → RQ2 visualizations
viz_rq3.py             → RQ3 visualizations

requirements.txt       → Python dependencies
README.md              → Project documentation
```

---

# 6. Requirements

Python 3.9+ recommended.

Major libraries:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- networkx  
- jupyter  

---

# 7. Notes

- Kaggle dataset must be downloaded separately.  
- The project has been cleaned of unused modules—only essential files remain.  
- The notebook and script both reproduce the full analysis pipeline.

---

Enjoy exploring NBA passing networks and team dynamics!
