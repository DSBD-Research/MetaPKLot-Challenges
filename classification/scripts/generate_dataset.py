from tqdm import tqdm
from glob import glob
import argparse
import json
import cv2
import os

def main():
    args = get_args()
    handle_input(args)
    uncompress_jsons(args.pklot)

    create_dataset(args.pklot, get_jsons_names(args.pklot, "pklot"), args.output + "PKLot/")
    create_dataset(args.pklot, get_jsons_names(args.pklot, "cnr"), args.output + "CNRPark-EXT/")
    create_dataset(args.plds, get_jsons_names(args.pklot, "plds"), args.output + "PLds/")

    print("Done.")

def create_dataset(images_dir, jsons_files, output_dir):
    for json_file in jsons_files:
        spots_json = json.load(open(json_file, "r"))

        subset_dir = get_subset_dir(json_file)

        if subset_dir.lower() == "vmlix/" or subset_dir.lower() == "vxusd/":
            subset_dir = "VXUSD,VMLIX/"

        put_annotations_inside_images(spots_json)

        for image_dict in tqdm(spots_json['images'], desc=f"Cropping images from {json_file}"):
            image = cv2.imread(f"{images_dir}/{image_dict['file_name']}")

            image_date = date_to_string(image_dict['date'])
            image_date_time = date_time_to_string(image_dict['date'], image_dict['time'])

            occupied_id = 0
            empty_id = 0

            os.makedirs(f"{output_dir}{subset_dir}{image_date}/occupied", exist_ok=True)
            os.makedirs(f"{output_dir}{subset_dir}{image_date}/empty", exist_ok=True)

            for annotation in image_dict['annotations']:
                best_rotated_rect = annotation['best_rotated_rect']
                occupance = annotation['category_id']

                center = best_rotated_rect[0]
                size = best_rotated_rect[1]
                angle = best_rotated_rect[2]

                M = cv2.getRotationMatrix2D(center, angle, 1)
                image_rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                image_cropped = cv2.getRectSubPix(image_rotated, [int(u) for u in size], center)

                if angle >= 45:
                    image_cropped = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)

                if occupance == 1:
                    cv2.imwrite(f"{output_dir}{subset_dir}{image_date}/occupied/{image_date_time}-{occupied_id}.jpg", image_cropped)
                    occupied_id += 1
                elif occupance == 0:
                    cv2.imwrite(f"{output_dir}{subset_dir}{image_date}/empty/{image_date_time}-{empty_id}.jpg", image_cropped)
                    empty_id += 1

def date_time_to_string(date, time):
    year, month, day = date
    hour, minute, second = time

    new_date = [str(year).zfill(4), str(month).zfill(2), str(day).zfill(2)]
    new_time = [str(hour).zfill(2), str(minute).zfill(2), str(second).zfill(2)]

    new_date = "_".join(str(x) for x in new_date)
    new_time = "_".join(str(x) for x in new_time)

    return f"{new_date}_{new_time}"

def date_to_string(date):
    year, month, day = date
    new_date = [str(year).zfill(4), str(month).zfill(2), str(day).zfill(2)]

    return "-".join(str(x) for x in new_date)

def put_annotations_inside_images(spots_json):
    images_idx = {image['id']: i for i, image in enumerate(spots_json['images'])}

    for annotation in spots_json['annotations']:

        image_idx = images_idx[annotation['image_id']]
        if spots_json['images'][image_idx].get('annotations') is None:
            spots_json['images'][image_idx]['annotations'] = [annotation]
        else:
            spots_json['images'][image_idx]['annotations'].append(annotation)

def get_subset_dir(json_file):
    json_file_lower = json_file.lower()

    if "ufpr04" in json_file_lower:
        return "UFPR04/"
    elif "ufpr05" in json_file_lower:
        return "UFPR05/"
    elif "pucpr" in json_file_lower:
        return "PUCPR/"
    elif "isshk" in json_file_lower:
        return "ISSHK/"
    elif "qridr" in json_file_lower:
        return "QRIDR/"
    elif "vmlix" in json_file_lower:
        return "VMLIX/"
    elif "vxusd" in json_file_lower:
        return "VXUSD/"
    elif "camera-1" in json_file_lower:
        return "CAMERA1/"
    elif "camera-2" in json_file_lower:
        return "CAMERA2/"
    elif "camera-3" in json_file_lower:
        return "CAMERA3/"
    elif "camera-4" in json_file_lower:
        return "CAMERA4/"
    elif "camera-5" in json_file_lower:
        return "CAMERA5/"
    elif "camera-6" in json_file_lower:
        return "CAMERA6/"
    elif "camera-7" in json_file_lower:
        return "CAMERA7/"
    elif "camera-8" in json_file_lower:
        return "CAMERA8/"
    elif "camera-9" in json_file_lower:
        return "CAMERA9/"

def uncompress_jsons(pklot_path):
    spots_path = get_spots(pklot_path)
    tar_files = glob(f"{spots_path}/**/*.tar.xz", recursive=True)

    for tar_file in tqdm(tar_files, desc="Uncompressing spots jsons"):
        os.system(f"tar -xf {tar_file} -C {os.path.dirname(tar_file)}")

def get_spots(pklot_path):
    return f"{pklot_path}annotations/original/spots"

def get_jsons_names(pklot_path, dataset):
    if dataset == "pklot":
        return [f"{get_spots(pklot_path)}/PKLot/ufpr04_spots.json", f"{get_spots(pklot_path)}/PKLot/ufpr05_spots.json",
                f"{get_spots(pklot_path)}/PKLot/pucpr_spots.json"]
    
    elif dataset == "cnr":
        return [f"{get_spots(pklot_path)}/CNRPark-EXT/cnr-camera-{i}_spots.json" for i in range(1, 10)]
    
    elif dataset == "plds":
        return [f"{get_spots(pklot_path)}/PLds/isshk_spots.json", f"{get_spots(pklot_path)}/PLds/qridr_spots.json",
                f"{get_spots(pklot_path)}/PLds/vmlix_spots.json", f"{get_spots(pklot_path)}/PLds/vxusd_spots.json"]

def handle_input(args):
    if not os.path.exists(args.pklot):
        print(f"Path to PKLot2.0 not found: {args.pklot}")
        exit(1)

    if not os.path.exists(args.plds):
        print(f"Path to PLds not found: {args.plds}")
        exit(1)

    if os.path.exists(args.output):
        print(f"Output path already exists: {args.output}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() == "n":
            exit()
        elif response.lower() == "y":
            os.system(f"rm -rf {args.output}")
            os.mkdir(args.output)
        else:
            print("Invalid response")
            exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--pklot", type=str, help="Path to PKLot2.0", required=True)
    parser.add_argument("--plds", type=str, help="Path to PLds generated by script", required=True)
    parser.add_argument("--output", type=str, help="Output path", required=True)
    args = parser.parse_args()

    args.pklot += "/" if not args.pklot.endswith("/") else ""
    args.plds += "/" if not args.plds.endswith("/") else ""
    args.output += "/dataset/" if not args.output.endswith("/") else "dataset/"

    return args

if __name__ == "__main__":
    main()
