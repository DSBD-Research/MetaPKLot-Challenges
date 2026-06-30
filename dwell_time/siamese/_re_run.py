import datetime
import argparse
import os

def main(args):
    assert os.path.exists(args.config), "Invalid config"

    begin_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    exp_name = f"rerun_{args.models.split('/')[-1]}"
    os.makedirs(f"_results/{exp_name}", exist_ok=True)
    os.makedirs("_summaries", exist_ok=True)

    print(f"\nExecuting experiment with pretrained models in {args.models}")
    pretrained_models = os.listdir(args.models)
    for model in pretrained_models:
        os.system(f"python3 _re_main.py -c {args.config} -m {os.path.join(args.models, model)} -s {exp_name}")

    end_datetime = "--".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" "))

    print("\nWriting summary")
    os.system(f"python3 summary.py -f _results/{exp_name} -bd {begin_datetime} -ed {end_datetime} -c {args.config}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Models training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--models", "-m", type=str, required=True)

    main(parser.parse_args())