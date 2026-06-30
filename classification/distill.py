import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os

from src.Model import Model

def main():
    args = get_args()
    num_executions = get_num_executions(args.train_dir)

    executions_distill = []

    if not args.augmentation_prob:
        args.augmentation_prob = 0.1

    if args.execution is None:
        for execution_idx in range(1, num_executions + 1):
            os.makedirs(args.train_dir + f"/execution{execution_idx}/distill/", exist_ok=True)
            n_distill = len(os.listdir(args.train_dir + f"/execution{execution_idx}/distill/"))
            os.makedirs(args.train_dir + f"/execution{execution_idx}/distill/distill{n_distill + 1}/", exist_ok=True)

            executions_distill.append(n_distill + 1)
    else:
        os.makedirs(args.train_dir + f"/execution{args.execution}/distill/", exist_ok=True)
        n_distill = len(os.listdir(args.train_dir + f"/execution{args.execution}/distill/"))
        os.makedirs(args.train_dir + f"/execution{args.execution}/distill/distill{n_distill + 1}/", exist_ok=True)

        executions_distill.append(n_distill + 1)
            

    for execution_idx in range(1, num_executions + 1):
        if args.execution is not None and args.execution != execution_idx:
            continue
        
        model_config_path = args.train_dir + f"/execution{execution_idx}/train/config.yaml"
        model_config = yaml.safe_load(open(model_config_path, "r"))

        handle_config(model_config)

        model = Model(model_config, "distill", args.gpu, args.pretrained_model)

        if args.execution is None:
            model.set_distill_output_path(args.train_dir + f"/execution{execution_idx}/distill/", 
                                          executions_distill[execution_idx - 1])
        else:
            model.set_distill_output_path(args.train_dir + f"/execution{execution_idx}/distill/", 
                                          executions_distill[0])
        model.save_distill_configs(args.train_days, args.val_days, args.threshold, args.student_model_type, args.student_grayscale)
        model.set_mean_std_json(args.train_dir + f"/execution{execution_idx}/train/mean_std.json")
        model.create_transformer()
        model.calculate_mean_std_student(args.student_model_type, args.student_grayscale)
        model.save_student_mean_std_json()
        model.create_student_transformer(args.student_model_type, args.use_data_augmentation, args.augmentation_prob)
        model.make_distill_dataloaders(args.train_days, args.val_days, args.student_grayscale)
        model.set_teacher_models_paths(args.train_dir + f"/execution{execution_idx}/train/models/")

        post_threshold_images_paths, post_threshold_targets = get_post_threshold_images_paths(model, args.threshold, args.ensemble_models)

        model.save_pseudo_targets(post_threshold_targets)

        pretrain_student_model(model, args.student_model_type, args.student_grayscale)

        train_student_models(model, post_threshold_images_paths, post_threshold_targets, args.student_model_type, args.student_grayscale)

        test_student_models(model, args.student_grayscale)

        test_student_base_model(model, args.student_grayscale)

def test_student_base_model(model, grayscale):
    thresholds = model.get_student_threshold_json()

    scaler = torch.amp.GradScaler()

    model.write_on_log(f"Start testing student base model... ")

    for loader_idx in range(model.get_distill_num_loaders()):
        model.set_dataloader(loader_idx)
        model.load_student_base_model(grayscale)
        model.set_student_model_to_eval()

        predictions = []
        gd_labels = []
        with torch.no_grad():
            for images, labels, _ in model.get_student_test_dataloader():
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict_student_model(images_)
                
                    confidences = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                predictions.extend(confidences)
                gd_labels.extend(labels_.cpu().numpy())
            
        threshold = thresholds["student"]

        model.save_test_student_base_model_json(predictions, gd_labels)
        model.save_single_metrics_base_distill(predictions, gd_labels, threshold)

    model.write_on_log("Done.")

def test_student_models(model, grayscale):
    scaler = torch.amp.GradScaler()

    thresholds = model.get_student_threshold_json()
    for loader_idx in range(model.get_distill_num_loaders()):
        model.set_dataloader(loader_idx)
        model.set_student_model(loader_idx, grayscale)
        model.set_student_model_to_eval()

        model.write_on_log(f"Start testing {model.get_student_model_name()}... ")

        predictions = []
        gd_labels = []
        with torch.no_grad():
            for images, labels, _ in model.get_student_test_dataloader():
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict_student_model(images_)
                    
                    confidences = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                predictions.extend(confidences)
                gd_labels.extend(labels_.cpu().numpy())
        
        threshold = thresholds[model.get_student_model_name()]

        model.save_test_student_model_json(predictions, gd_labels)
        model.save_single_metrics_distill(predictions, gd_labels, threshold)

        model.write_on_log("Done.")

def train_student_models(model, post_threshold_images_paths, post_threshold_targets, student_base_model_type, grayscale):
    scaler = torch.amp.GradScaler()

    for loader_idx in range(model.get_distill_num_loaders()):
        model.set_dataloader(loader_idx)
        model.set_student_base_model(student_base_model_type, grayscale)

        model.write_on_log(f"Start training {model.get_student_model_name()}...")
        model.write_on_log(f"Training with {len(post_threshold_images_paths[loader_idx])} images")

        model.create_post_threshold_distill_train_dataloader(post_threshold_images_paths[loader_idx], grayscale)
        
        optimizer = optim.Adam(model.get_student_model().parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        best_validation_loss = float("inf")
        best_train_loss = float("inf")
        best_y = None
        best_y_prob = None
        for epoch in range(model.get_num_epochs()):
            model.set_student_model_to_train()

            model.write_on_log(f"Epoch {epoch + 1}/{model.get_num_epochs()}")

            # Training
            total_loss = 0
            for images, _, images_paths in model.get_post_threshold_train_dataloader():
                images_ = images.to(model.get_device())
 
                optimizer.zero_grad()

                pseudo_labels = [post_threshold_targets[loader_idx][image_path] for image_path in images_paths]
                pseudo_labels = torch.tensor(pseudo_labels).to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict_student_model(images_)

                    loss = criterion(outputs, pseudo_labels)

                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            training_loss = total_loss / len(model.get_post_threshold_train_dataloader())

            # Validation
            model.set_student_model_to_eval()
            correct_predictions = 0
            total_predictions = 0
            total_loss = 0
            y = []
            y_prob = []

            with torch.no_grad():
                for images, labels, _ in model.get_student_val_dataloader():
                    images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                    with torch.amp.autocast("cuda"):
                        outputs = model.predict_student_model(images_)
                        loss = criterion(outputs, labels_)
                    total_loss += loss.item()

                    predicted = torch.max(outputs.data, 1)[1]
                    total_predictions += labels_.size(0)
                    correct_predictions += (predicted == labels_).sum().item()

                    y.extend(labels_.cpu().numpy())
                    y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            if len(model.get_student_val_dataloader()) != 0:
                validation_loss = total_loss / len(model.get_student_val_dataloader())
                validation_accuracy = correct_predictions / total_predictions

            model.write_on_log(f"Training loss: {training_loss}")

            if len(model.get_student_val_dataloader()) != 0:
                model.write_on_log(f"Validation loss: {validation_loss}")
                model.write_on_log(f"Validation accuracy: {validation_accuracy}")
            
            if len(model.get_student_val_dataloader()) != 0:
                if validation_loss < best_validation_loss:
                    model.write_on_log(f"{validation_loss} < {best_validation_loss}. Saving model...")

                    best_validation_loss = validation_loss
                    model.save_student_model()

                    best_y = y.copy()
                    best_y_prob = y_prob.copy()
            else:
                if training_loss < best_train_loss:
                    model.write_on_log(f"{training_loss} < {best_train_loss}. Saving model...")

                    best_train_loss = training_loss
                    model.save_student_model()

                    best_y = y.copy()
                    best_y_prob = y_prob.copy()

            model.write_on_log("")
        
        if len(model.get_student_val_dataloader()) != 0:
            eer_threshold = plot_roc_curve(best_y, best_y_prob, model.get_student_model_name(), model.get_distill_output_path())
            model.save_student_eer_threshold(eer_threshold)
        else:
            model.save_student_eer_threshold(0.5)

def plot_roc_curve(y, y_prob, model_name, model_output_dir):
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    plt.plot(fpr, tpr, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {model_name}')
    os.makedirs(f"{model_output_dir}plots/", exist_ok=True)
    plt.savefig(f"{model_output_dir}plots/{model_name}_roc_curve.png")
    plt.clf()

    eer_threshold = compute_eer_threshold(fpr, tpr, thresholds)

    return eer_threshold

def compute_eer_threshold(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)

    return thresholds[min_index]

def pretrain_student_model(model, student_model_type, grayscale):
    model.write_on_log(f"Start pretraining student model...")

    scaler = torch.amp.GradScaler()

    model.load_student_model_type(student_model_type, grayscale)

    optimizer = optim.Adam(model.get_student_model_parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    best_train_loss = float("inf")
    best_y = None
    best_y_prob = None
    for epoch in range(model.get_num_epochs()):
        model.set_student_model_to_train()

        model.write_on_log(f"Epoch {epoch + 1}/{model.get_num_epochs()}")

        # Training
        total_loss = 0
        for images, labels, _ in model.get_student_pretrain_train_dataloader():
            images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model.predict_student_model(images_)
                loss = criterion(outputs, labels_)

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        training_loss = total_loss / len(model.get_student_pretrain_train_dataloader())

        # Validation
        model.set_student_model_to_eval()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        y = []
        y_prob = []

        with torch.no_grad():
            for images, labels, _ in model.get_student_pretrain_val_dataloader():
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict_student_model(images_)
                    loss = criterion(outputs, labels_)
                total_loss += loss.item()

                predicted = torch.max(outputs.data, 1)[1]
                total_predictions += labels_.size(0)
                correct_predictions += (predicted == labels_).sum().item()

                y.extend(labels_.cpu().numpy())
                y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_dataloader_tam = len(model.get_student_pretrain_val_dataloader())

        if val_dataloader_tam != 0:
            validation_loss = total_loss / val_dataloader_tam
            validation_accuracy = correct_predictions / total_predictions

        model.write_on_log(f"Training loss: {training_loss}")

        if val_dataloader_tam != 0:
            model.write_on_log(f"Validation loss: {validation_loss}")
            model.write_on_log(f"Validation accuracy: {validation_accuracy}")

        if val_dataloader_tam != 0:
            if validation_loss < best_validation_loss:
                model.write_on_log(f"{validation_loss} < {best_validation_loss}. Saving model...")

                best_validation_loss = validation_loss
                model.save_student_base_model()

                best_y = y.copy()
                best_y_prob = y_prob.copy()
        else:
            if training_loss < best_train_loss:
                model.write_on_log(f"{training_loss} < {best_train_loss}. Saving model...")

                best_train_loss = training_loss
                model.save_student_base_model()

                best_y = y.copy()
                best_y_prob = y_prob.copy()

        model.write_on_log("")
    
    eer_threshold = plot_roc_curve(best_y, best_y_prob, model.get_student_model_name(), model.get_distill_output_path())
    model.save_student_base_eer_threshold(eer_threshold)

def get_post_threshold_images_paths(model, threshold, ensemble_models):
    model.write_on_log("Getting post threshold images...")
    post_threshold_images_paths = []
    post_threshold_targets = []

    for loader_idx in range(model.get_distill_num_loaders()):
        model.set_dataloader(loader_idx)
        post_threshold_images_paths.append([])
        post_threshold_targets.append({})

        image_path_to_confidence = {}

        if ensemble_models == "all":
            for model_idx in range(model.get_n_teacher_distill_models()):
                model.load_distill_teacher_model(model_idx)

                model.set_to_eval()

                with torch.no_grad():
                    for images, _, images_paths in model.get_student_train_dataloader():
                        images_ = images.to(model.get_device())

                        with torch.amp.autocast("cuda"):
                            outputs = model.predict(images_)

                        confidences = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                        for image_path, confidence in zip(images_paths, confidences):
                            if image_path in image_path_to_confidence:
                                image_path_to_confidence[image_path] += confidence
                            else:
                                image_path_to_confidence[image_path] = confidence
        
        elif ensemble_models == "total":
            model.load_distill_total_teacher_model()

            model.set_to_eval()

            with torch.no_grad():
                for images, _, images_paths in model.get_student_train_dataloader():
                    images_ = images.to(model.get_device())

                    with torch.amp.autocast("cuda"):
                        outputs = model.predict(images_)

                    confidences = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                    for image_path, confidence in zip(images_paths, confidences):
                        if image_path in image_path_to_confidence:
                            image_path_to_confidence[image_path] += confidence
                        else:
                            image_path_to_confidence[image_path] = confidence

        n_models = model.get_n_teacher_distill_models() if ensemble_models == "all" else 1

        for key in image_path_to_confidence.keys():
            image_path_to_confidence[key] /= n_models

            if image_path_to_confidence[key] >= threshold or image_path_to_confidence[key] <= 1 - threshold:
                post_threshold_images_paths[loader_idx].append(key)

                target = 1 if image_path_to_confidence[key] >= 0.5 else 0
                post_threshold_targets[loader_idx][key] = target

    return post_threshold_images_paths, post_threshold_targets

def handle_config(config):
    config["datasets_path"] += "/" if not config["datasets_path"].endswith("/") else ""
    config["output_dir"] += "/" if not config["output_dir"].endswith("/") else ""

def get_num_executions(train_dir):
    return len(os.listdir(train_dir))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to the train dir", required=True)
    parser.add_argument("--train_days", type=int, help="Number of days used for operation", required=True)
    parser.add_argument("--val_days", type=int, help="Number of days used for operation", required=True)
    parser.add_argument("--threshold", type=float, help="Threshold of confidence on teacher classifications", required=True)
    parser.add_argument("--student_model_type", type=str, help="Student model name", required=True)
    parser.add_argument("--student_grayscale", type=str, help="Use grayscale to train student model. True or False", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)
    parser.add_argument("--ensemble_models", type=str, help="All or total", required=True)
    parser.add_argument("--execution", type=int, help="Execution number", required=False)
    parser.add_argument("--pretrained_model", type=str, help="Path to the pretrained model", required=False)
    parser.add_argument("--use_data_augmentation", type=str, help="True or False", required=True)
    parser.add_argument("--augmentation_prob", type=float, help="augmentation prob", required=False)

    args = parser.parse_args()

    args.train_dir += "/" if not args.train_dir.endswith("/") else ""

    return args

if __name__ == "__main__":
    main()
