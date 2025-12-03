# NBA_SI-608

Quick start to run the pipeline in `master.py`:

1) (Optional) Create and activate a virtual environment.  
2) Install dependencies: `pip install -r requirements.txt`  
3) Place the play-by-play CSVs in a folder named `NBA-Data` in the project root.  
4) From the project root, run: `python master.py`

The script will read all CSVs from `NBA-Data`, compute the metrics, and generate the figures. You do not need to open or inspect the large data files to run it.
