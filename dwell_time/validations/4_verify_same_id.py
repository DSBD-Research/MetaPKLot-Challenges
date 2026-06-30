from collections import defaultdict
import subprocess
import argparse

def main(args):
    command = f"find {args.dataset} -type f -iname '*.jpg'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    output = result.stdout.strip().split("\n")
    treated_output = [(split[-1].split("#")[0], split[-1].split("#")[1].replace(".jpg", ""), split[-2], split[-3]) for split in [line.split("/") for line in output]]

    spots_by_car = defaultdict(list)
    days_by_car_and_spot = defaultdict(list)
    subsets_by_car = defaultdict(list)

    for car_id, day, spot_id, subset in treated_output:
        spots_by_car[car_id].append(spot_id)
        days_by_car_and_spot[car_id + spot_id].append(day)
        subsets_by_car[car_id].append(subset)

    spots_by_car = {car_id: set(set(values)) for car_id, values in spots_by_car.items()}
    days_by_car_and_spot = {id: set(set(values)) for id, values in days_by_car_and_spot.items()}
    subsets_by_car = {car_id: set(set(values)) for car_id, values in subsets_by_car.items()}

    shared_ids = { car_id: values for car_id, values in spots_by_car.items() if len(values) > 1 }
    for id, spot_ids in shared_ids.items():
        print(f"Car: {id} - Subset: {subsets_by_car[id]}")

        for spot_id in spot_ids:
            print(f"  Spot: {spot_id}")
            days = days_by_car_and_spot[id + spot_id]
            print(f"    Days: {', '.join(sorted(days))};")
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="verify", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", "-ds", type=str, required=True)

    main(parser.parse_args())

# Verify if more than one different car has te same id
# Use the spot_id to check it (if the same ID appers in more than one spot, its duplicated)
# can miss some duplicates, but it's ok
# The dataset path passed as argument needs to be the one generated bu the script 4_save_cars.py