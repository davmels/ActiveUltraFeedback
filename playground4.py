from datasets import load_from_disk 
dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_maxmin_dpo")
print(dataset)
print(dataset["chosen"][0])
print("-----------------------------------------\n\n\n")
print(dataset["rejected"][0])
