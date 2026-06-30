import datetime
import argparse
import uuid
import os

def main(args):
    assert os.path.exists(args.config), "Invalid config"

    exp_uuid = str(uuid.uuid4()).replace('-', '')[:5]
    exp_name = f"{args.config.split("/")[1].split(".")[0]}_{exp_uuid}"

    begin_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    os.makedirs(f"_results/{exp_name}", exist_ok=True)
    os.makedirs(f"_models/{exp_name}", exist_ok=True)
    os.makedirs("_summaries", exist_ok=True)

    print("\nExecuting experiment")
    os.system(f"python3 main.py -c {args.config} -s {exp_name}")
    os.system(f"python3 main.py -c {args.config} -s {exp_name}")
    os.system(f"python3 main.py -c {args.config} -s {exp_name}")
    os.system(f"python3 main.py -c {args.config} -s {exp_name}")
    os.system(f"python3 main.py -c {args.config} -s {exp_name}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{exp_name} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Models training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())