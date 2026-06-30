import datetime
import argparse
import uuid
import json
import os

def main(args):
    assert os.path.exists(args.config), "Invalid config"
    with open(args.config, "r") as file:
        config = json.load(file)

    exp_uuid = str(uuid.uuid4()).replace('-', '')[:5]
    exp_name = f"{args.config.split('/')[1].split('.')[0]}_{exp_uuid}"

    classif_ensembles = os.listdir(config["models"]["classification"]["path"])
    siamese_models = [file.replace(".json", "") for file in os.listdir(config["models"]["siamese"]["path"]) if ".json" in file]

    os.makedirs(f"_results/{exp_name}", exist_ok=True)
    os.makedirs("_summaries", exist_ok=True)

    begin_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    print("\nExecuting experiment")
    for index in range(len(classif_ensembles)):
        os.system(f"python3 main.py -c {args.config} -ce {classif_ensembles[index]} -sm {siamese_models[index]} -s {exp_name}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{exp_name} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())
