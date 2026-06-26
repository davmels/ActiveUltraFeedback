from datasets import load_from_disk

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_maxmin_dpo_refiltered")
 
print(dataset)

dataset2 = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_maxmin_dpo")

print(dataset2)