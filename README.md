# ijcnn-251
The technical supplementary is named ijcnn2022_supp.pdf.

The codes for replicating benchmark datasets, IHDP experiments and Twins experiments, are divided into four files.

The 1000 IHDP datasets can be downloaded from https://www.fredjo.com/

The Twins dataset can be downloaded from https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS. 
Then open "generate_twins_data.py" to generate 100 Twins datasets.

When you run general models such as Lasso+LR, you should open "EXP/general" (e.g., EXP is IHDP or Twins) and run "run_general_EXP.py".
When you run TARNET or Dragonnet, you should open "EXP/NET" (e.g., EXP is IHDP or Twins) and run "run_ocnet_EXP.py".

When the running script finishes, you can run compute_results_general.py for file "general" (compute_results_general.py for file "NET", resp.) to produce the final result.

We upload the whole experiment results reported in our paper for your reference. They are: IHDP_general_epsilonATE_list_0-1000_test.csv; IHDP_NET_epsilonATE_list_0-1000_test.csv; Twins_general_epsilonATE_list_0-100_test.csv; Twins_NET_epsilonATE_list_0-100_test.csv.
