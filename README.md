# ijcnn-251
The technical supplementary is named ijcnn2022_supp.pdf.

The codes for replicating benchmark datasets, IHDP experiments and Twins experiments, are divided into four files.

When you run general models such as Lasso+LR, you should open "EXP/general" (e.g., EXP is IHDP or Twins) and run "run_general_EXP.py".
When you run TARNET or Dragonnet, you should open "EXP/NET" (e.g., EXP is IHDP or Twins) and run "run_ocnet_EXP.py".

The 1000 IHDP datasets can be downloaded from https://www.fredjo.com/

The Twins dataset can be downloaded from https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS. 
Then open "generate_twins_data.py" to generate 100 Twins datasets.
