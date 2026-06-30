import json
import os
import sys
from statistics import mean, stdev

def main():
    train_folder, distill_n = get_train_folder()

    n_executions = len(os.listdir(train_folder))

    accuracies = []
    totals = []

    for execution_idx in range(n_executions):
        distill_path = os.path.join(train_folder, f"execution{execution_idx + 1}", "distill", f"distill{distill_n}", "pseudo_targets.json")

        total = 0
        correct = 0

        json_file = json.load(open(distill_path, "r"))

        for dic in json_file:
            for key in dic:
                total += 1
                
                occupance = key.split('/')[-2]

                if dic[key] == 1 and occupance == "occupied":
                    correct += 1
                elif dic[key] == 0 and occupance == "empty":
                    correct += 1
        
        accuracies.append(correct / total)
        totals.append(total)

    print(f"Mean: {(mean(accuracies) * 100):.2f}")
    print(f"Stdev: {(stdev(accuracies) * 100):.2f}")
    print(f"Mean total of images for training: {sum(totals) / len(totals)}")

def get_train_folder():
    if len(sys.argv) != 3:
        print("Usage: python calculate_accuracy_psudolabels.py <train_folder> <distill_n>")
        sys.exit(1)

    return sys.argv[1], int(sys.argv[2])

if __name__ == "__main__":
    main()
