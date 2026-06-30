import argparse
import faster_coco_eval

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools import mask as mask_utils
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import os

from torch.nn.functional import softmax, sigmoid
from sklearn.metrics import average_precision_score

from spot import save_prediction_to_graph_data, save_predictions
from torch_geometric.data import HeteroData
import torch
import numpy as np
from collections import defaultdict
import numpy as np
import torch
import math
from math import pi, sin, cos
from datetime import date as _date
import torch.nn.functional as F
import numpy as np
from pycocotools import mask as maskUtils
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import NeighborLoader

from torch_geometric.nn import (
    GraphConv,
    HeteroConv,
    Linear,
    GATConv,
    GeneralConv,
    TransformerConv,
    FiLMConv,
    SAGEConv,
    GATv2Conv,
    GCNConv,
)
from torch_geometric.nn import norm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import aggr

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import json
import torch
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from os.path import join
from spot import filter_spot_gt

from non_representative_days import get_non_representative_days


class LogitBlendHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.linear = Linear(2, 1)

        # Initialize parameters
        torch.nn.init.constant_(self.linear.weight, 1)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, gnn_logit, prior_prob, eps=1e-5):
        gnn_logit = gnn_logit.unsqueeze(-1) if gnn_logit.dim() == 1 else gnn_logit
        prior_prob = prior_prob.clamp(eps, 1 - eps)
        prior_logit = torch.log(prior_prob) - torch.log1p(-prior_prob)
        final_logit = self.beta * gnn_logit + self.alpha * prior_logit + self.bias
        return final_logit


class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=1):
        super().__init__()
        depth = 2  # Number of layers, can be adjusted
        # First and second HeteroConv layers
        self.convs = nn.ModuleList()
        self.post = nn.ModuleList()

        self.in_proj = nn.ModuleDict(
            {
                "spot": Linear(1, hidden_channels),
                # "image": Linear(-1, hidden_channels),
                "car": Linear(256, hidden_channels),
            }
        )

        for _ in range(depth):
            self.convs += [
                HeteroConv(
                    {
                        ("car", "inside", "spot"): SAGEConv(
                            (hidden_channels, hidden_channels),
                            hidden_channels,
                            aggr=["add", "mean"],
                        ),
                        ("spot", "rev_inside", "car"): SAGEConv(
                            (hidden_channels, hidden_channels),
                            hidden_channels,
                            aggr=["add", "mean"],
                        ),
                    },
                    aggr="cat",
                )
            ]

        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: norm.LayerNorm(hidden_channels, affine=True)
                        for node_type in metadata[0]
                    }
                )
                for _ in range(depth)
            ]
        )

        self.lin = nn.ModuleDict(
            {
                node_type: Linear(hidden_channels, out_channels)
                for node_type in metadata[0]
            }
        )
        self.dropout = 0.1

        self.logit_blend = LogitBlendHead()

        self.reset_parameters()  # ensure fresh init when you construct

    def reset_parameters(self):
        # Convs
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        # Norms
        for layer_norms in self.norms:
            for m in layer_norms.values():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        # Output heads
        for head in self.lin.values():
            if hasattr(head, "reset_parameters"):
                head.reset_parameters()

    def forward(self, x_dict, edge_index_dict, prior_score):
        x_dict = {nt: self.in_proj[nt](x) for nt, x in x_dict.items()}

        for layer, conv in enumerate(self.convs):
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }
            out_dict = conv(x_dict, edge_index_dict)
            for key, x in out_dict.items():
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
                x = self.norms[layer][key](x)
                x_dict[key] = x

        # Output layer only for nodes that are present in x_dict
        out = {}
        for node_type in self.lin:
            if node_type in x_dict:
                out[node_type] = self.lin[node_type](x_dict[node_type])
                if node_type == "spot":
                    out[node_type] = self.logit_blend(
                        out[node_type].view(-1), prior_score
                    )
                    out[node_type] = out[node_type].view(-1, 1)  # ensure shape [N, 1]
        return out


def ap50_evaluation(root, pos_spots, val_subsets, labelled_spots_dir):
    aps = list()

    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        for subset in val_subsets:
            print(f"{30 * '*'} Processing subset: {subset} {30 * '*'}")
            spots_input = f"{root}{subset}_spots.json"

            with open(spots_input, "r") as f:
                spots_data = json.load(f)

            non_representative_days = get_non_representative_days(subset)
            spots_data = filter_spot_gt(spots_data, non_representative_days)
            if "info" not in spots_data:
                spots_data["info"] = {}

            output_labelled_spots_file = join(
                labelled_spots_dir, f"{subset}_labelled_spots.json"
            )

            with open(output_labelled_spots_file, "r") as f:
                labelled_spots_predictions = json.load(f)

            # Load ground truth COCO object
            coco_gt = COCO()
            coco_gt.dataset = spots_data
            coco_gt.createIndex()

            # Filter predicitons based on pos_spots
            gnn_spots_predictions = []

            for pred in labelled_spots_predictions:
                pred_copy = deepcopy(pred)
                if pred["spot_pred_id"] in pos_spots:
                    pred_copy["score"] = pos_spots[pred["spot_pred_id"]]
                gnn_spots_predictions.append(pred_copy)

            for pred in labelled_spots_predictions:
                pred["category_id"] = 1  # Set category_id to 1 for GNN predictions

            for pred in gnn_spots_predictions:
                pred["category_id"] = 1  # Set category_id to 1 for GNN predictions

            coco_dt = coco_gt.loadRes(gnn_spots_predictions)

            coco_eval = COCOeval(
                coco_gt, coco_dt, iouType="segm"
            )  # or 'bbox' if bounding boxes only
            coco_eval.params.maxDets = [
                10,
                100,
                1000,
            ]  # instead of default [1, 10, 100]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            aps += [coco_eval.stats[1]]  # AP50 is at index 1

    mean_ap50 = np.mean(aps) if aps else 0.0
    return mean_ap50


def centroid_from_rle_decode(rle):
    """
    Returns (cx, cy) in image coords (x=column, y=row).
    If you want pixel-center coordinates, add +0.5 to each.
    """
    m = maskUtils.decode(rle)  # HxW uint8, Fortran-ordered (column-major)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None  # empty mask
    cx = xs.mean()
    cy = ys.mean()
    return float(cx), float(cy)


# --- 1) Build day nodes (unique per calendar day) ---
def _to_datestr(d):
    if isinstance(d, str):
        return d
    # assume list/tuple like [YYYY, MM, DD]
    y, m, dd = d
    return f"{int(y):04d}-{int(m):02d}-{int(dd):02d}"


def load_spot_dataset(data_dict):
    data = HeteroData()
    # ===========================================================================================================
    # =============================================== IMAGE NODES ===============================================
    # ===========================================================================================================

    image_id_map = {img["id"]: idx for idx, img in enumerate(data_dict["image_nodes"])}
    car_id_map = {car["id"]: idx for idx, car in enumerate(data_dict["car_nodes"])}
    car_by_id = {car["id"]: car for car in data_dict["car_nodes"]}

    image_to_embeddings = defaultdict(list)

    # ============================================= car FROM image (edge) ============================================
    edge_index_car_image = [[], []]  # [car_idx, image_idx]
    rotated_bboxes = []

    for edge in tqdm(
        data_dict["car_from_image"], desc="Processing car_from_image edges"
    ):
        if edge["image_id"] not in image_id_map or edge["car_id"] not in car_id_map:
            continue
        car_idx = car_id_map[edge["car_id"]]
        image_idx = image_id_map[edge["image_id"]]

        edge_index_car_image[0].append(car_idx)
        edge_index_car_image[1].append(image_idx)
        rotated_bboxes.append(edge["rotated_bbox"])
        car = car_by_id[edge["car_id"]]
        car["rotated_bbox"] = edge["rotated_bbox"]
        image_to_embeddings[edge["image_id"]].append(car["embeddings"])

    image_features = []
    image_ids = []
    for img in tqdm(data_dict["image_nodes"], desc="Processing image nodes"):
        image_features.append(img["embeddings"])
        image_ids.append(img["id"])

    # =========================================================================================================
    # =============================================== DAY NODES ===============================================
    # =========================================================================================================

    # Collect unique day keys from your already-built image_nodes
    day_keys = []
    for img in data_dict["image_nodes"]:
        dk = _to_datestr(img["date"])
        day_keys.append(dk)

    unique_days = sorted(set(day_keys))
    day_key_to_idx = {dk: i for i, dk in enumerate(unique_days)}

    # --- 2) Map image ids to their node indices (order in data["image"]) ---

    image_id_to_idx = {int(iid): idx for idx, iid in enumerate(image_ids)}

    # --- 3) Build edges: image -> day (and optionally reverse) ---
    img_to_day_src = []
    img_to_day_dst = []

    num_images_per_day = defaultdict(int)

    for img in data_dict["image_nodes"]:
        iid = int(img["id"])
        img_idx = image_id_to_idx[iid]
        dk = _to_datestr(img["date"])
        day_idx = day_key_to_idx[dk]

        img_to_day_src.append(img_idx)
        img_to_day_dst.append(day_idx)

        num_images_per_day[day_idx] += 1

    # ========= SPOT -> DAY edges (direct from spot.date) =========
    spot_id_map = {spot["id"]: idx for idx, spot in enumerate(data_dict["spot_nodes"])}

    spot_to_day_src, spot_to_day_dst = [], []
    for cur_spot in data_dict["spot_nodes"]:
        s_idx = spot_id_map[cur_spot["id"]]
        dk = _to_datestr(cur_spot["date"])
        if dk not in day_key_to_idx:
            continue  # skip spots with unknown dates
        d_idx = day_key_to_idx[dk]
        spot_to_day_src.append(s_idx)
        spot_to_day_dst.append(d_idx)

    # ==========================================================================================================
    # =============================================== CAR NODES ===============================================
    # ==========================================================================================================

    car_features = []
    car_ids = []

    for car in tqdm(data_dict["car_nodes"], desc="Processing car nodes"):
        car_features.append(car["embeddings"])  # assuming list or array of floats
        car_ids.append(car["id"])

    # ==========================================================================================================
    # =============================================== SPOT NODES ===============================================
    # ==========================================================================================================

    spot_id_map = {spot["id"]: idx for idx, spot in enumerate(data_dict["spot_nodes"])}
    spot_features = [
        [1 - (1.0 / spot["score_stats"][4])] for spot in data_dict["spot_nodes"]
    ]

    # === car INSIDE spot (edge) ===
    edge_index_car_spot = [[], []]
    edge_score_car_spot = []
    car_id_to_score = {car["id"]: car["score"] for car in data_dict["car_nodes"]}
    spot_id_to_score = {spot["id"]: spot["score"] for spot in data_dict["spot_nodes"]}

    for rel in tqdm(
        data_dict["car_inside_spot"], desc="Processing car_inside_spot edges"
    ):
        if rel["car_id"] not in car_id_map or rel["spot_id"] not in spot_id_map:
            continue
        car_idx = car_id_map[rel["car_id"]]
        spot_idx = spot_id_map[rel["spot_id"]]
        edge_index_car_spot[0].append(car_idx)
        edge_index_car_spot[1].append(spot_idx)
        edge_score_car_spot.append(
            [car_id_to_score[rel["car_id"]] + spot_id_to_score[rel["spot_id"]]]
        )

    data["car"].x = F.normalize(
        torch.tensor(car_features, dtype=torch.float), p=2, dim=-1
    )
    data["car"].score = torch.tensor(
        [car["score"] for car in data_dict["car_nodes"]], dtype=torch.float
    )
    data["car"].id = torch.tensor(car_ids, dtype=torch.long)

    data["spot"].x = torch.tensor(spot_features, dtype=torch.float)
    data["spot"].score = torch.tensor(
        [[spot["score"]] for spot in data_dict["spot_nodes"]], dtype=torch.float
    )
    data["spot"].y = torch.tensor(
        [[spot["label"]] for spot in data_dict["spot_nodes"]], dtype=torch.float
    )
    data["spot"].id = torch.tensor(
        [spot["id"] for spot in data_dict["spot_nodes"]], dtype=torch.long
    )

    data["car", "inside", "spot"].edge_index = torch.tensor(
        edge_index_car_spot, dtype=torch.long
    )

    spot_y = data["spot"].y.view(-1)  # [num_spots], float {0,1}
    device = spot_y.device

    # Cars connected to any positive spot become 1
    pos_spot_mask = spot_y > 0.5  # bool per spot
    pos_edges_mask = pos_spot_mask[
        data["car", "inside", "spot"].edge_index[1]
    ]  # bool per edge (car->spot)
    pos_car_idx = data["car", "inside", "spot"].edge_index[
        0, pos_edges_mask
    ]  # cars that touch a positive spot

    car_y = torch.zeros(data["car"].num_nodes, dtype=torch.float, device=device)
    if pos_car_idx.numel() > 0:
        car_y[pos_car_idx] = 1.0

    data["car"].y = car_y.unsqueeze(1)  # shape [num_cars, 1]

    data = ToUndirected()(data)
    return data


def evaluate(model, root, test_loader, device, labelled_spots_dir, val_subsets):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    spots = dict()
    all_preds = []
    all_targets = []

    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x_dict, batch.edge_index_dict, batch["spot"].score)[
                "spot"
            ]
            probs = sigmoid(out)  # probability scores
            pred = (probs > 0.5).long()  # binary predictions
            target = batch["spot"].y.long()
            spot_id = batch["spot"].id.long().tolist()

            cur_spots = {spot_id[i]: probs[i].cpu().item() for i in range(len(spot_id))}
            spots.update(cur_spots)

            # For average precision calculation
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # Confusion matrix counts
            tp += ((pred == 1) & (target == 1)).sum().item()
            tn += ((pred == 0) & (target == 0)).sum().item()
            fp += ((pred == 1) & (target == 0)).sum().item()
            fn += ((pred == 0) & (target == 1)).sum().item()

            correct += (pred == target).sum().item()
            total += target.size(0)

    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    ap = ap50_evaluation(root, spots, val_subsets, labelled_spots_dir)

    print(
        f"Test F1 Score: {f1_score:.4f} AP50: {ap:.4f} ACC: {acc:.4f} TP: {tp} TN: {tn} FP: {fp} FN: {fn}"
    )
    return spots


def split_decay(params):
    """Return (decay, no_decay) param lists: no WD for biases & norms."""
    decay, no_decay = [], []
    for n, p in params:
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "norm" in n.lower() or "layernorm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def margin_ranking_loss(logits, targets, margin=0.5):
    rank_loss_fn = torch.nn.MarginRankingLoss(margin=margin)

    tp_scores = logits[targets == 1]
    fp_scores = logits[targets == 0]

    if tp_scores.numel() > 0 and fp_scores.numel() > 0:
        n = min(tp_scores.size(0), fp_scores.size(0))
        tp_sample = tp_scores[:n]
        fp_sample = fp_scores[:n]

        target_rank = torch.ones(n, device=logits.device)
        rank_loss = rank_loss_fn(tp_sample, fp_sample, target_rank)
    else:
        rank_loss = None

    return rank_loss


def main(args):
    graph_dir = join(args.output_dir, "graphs", args.model, args.exp)
    weight_root = join(args.output_dir, "gnn_weights", args.model, args.exp)
    labelled_spots_dir = join(args.output_dir, "labelled_spots", args.model, args.exp)

    training_subsets = [
        "cnr-camera-1",
        "cnr-camera-2",
        "cnr-camera-3",
        "cnr-camera-4",
        "cnr-camera-5",
        "cnr-camera-6",
        "cnr-camera-7",
        "cnr-camera-8",
        "cnr-camera-9",
        "pucpr",
        "ufpr04",
        "ufpr05",
        "isshk",
        "qridr",
        "vmlix-vxusd",
    ]

    if "wo_cnr" in labelled_spots_dir:
        val_subsets = [
            "cnr-camera-1",
            "cnr-camera-2",
            "cnr-camera-3",
            "cnr-camera-4",
            "cnr-camera-5",
            "cnr-camera-6",
            "cnr-camera-7",
            "cnr-camera-8",
            "cnr-camera-9",
        ]
    elif "wo_pklot" in labelled_spots_dir:
        val_subsets = ["pucpr", "ufpr04", "ufpr05"]
    elif "wo_plds" in labelled_spots_dir:
        val_subsets = ["isshk", "qridr", "vmlix-vxusd"]
    else:
        raise ValueError("Please define validation subsets")

    for val_subset in val_subsets:
        if val_subset in training_subsets:
            training_subsets.remove(val_subset)

    # Load graph structures for training subsets

    training_data = None

    for subset in training_subsets:
        graph_file = join(graph_dir, f"{subset}_graph.json")

        with open(graph_file, "r") as f:
            graph_structure = json.load(f)

        if training_data is None:
            training_data = graph_structure
        else:
            for key in training_data.keys():
                training_data[key].extend(graph_structure[key])

    val_data = None

    for subset in val_subsets:
        graph_file = join(graph_dir, f"{subset}_graph.json")

        with open(graph_file, "r") as f:
            graph_structure = json.load(f)

        if val_data is None:
            val_data = graph_structure
        else:
            for key in val_data.keys():
                val_data[key].extend(graph_structure[key])

    train_subset = load_spot_dataset(training_data)
    val_subset = load_spot_dataset(val_data)

    train_subset.metadata()
    val_subset.metadata()

    num_neighbors = {
        ("car", "inside", "spot"): [10, 10, 10],
        ("spot", "rev_inside", "car"): [10, 10, 10],
    }

    train_loader = NeighborLoader(
        train_subset,
        input_nodes=("spot", torch.arange(len(train_subset["spot"].y))),
        num_neighbors=num_neighbors,  # 2-layer sampling
        batch_size=256,
        shuffle=True,
    )

    test_loader = NeighborLoader(
        val_subset,
        input_nodes=("spot", torch.arange(len(val_subset["spot"].y))),
        num_neighbors={key: (-1, -1, -1) for key in num_neighbors},  # full neighbors
        batch_size=256,
        shuffle=False,
    )

    # ----- 1) Group parameters by role -----
    # Encoder = everything except the final linear heads and the logit blend head
    encoder_named = []
    convs_named = []
    head_named = []
    blend_named = []
    norm_named = []

    scheduler = None
    device = torch.device("cuda")
    model = HeteroGNN(train_subset.metadata(), hidden_channels=512).to(device)

    for n, p in model.named_parameters():
        if n.startswith("lin."):  # per-type heads (car/spot)
            head_named.append((n, p))
        elif n.startswith("logit_blend."):  # prior fusion head
            blend_named.append((n, p))
        elif n.startswith("convs."):
            convs_named.append((n, p))  # GNN conv layers
        elif n.startswith("norm."):
            norm_named.append((n, p))  # Normalization layers
        else:
            encoder_named.append((n, p))  # in_proj, etc.

    enc_decay, enc_no_decay = split_decay(encoder_named)
    head_decay, head_no_decay = split_decay(head_named)
    blend_decay, blend_no_decay = split_decay(blend_named)
    convs_decay, convs_no_decay = split_decay(convs_named)
    norm_decay, norm_no_decay = split_decay(norm_named)

    enc_max_lr = 1e-5
    head_max_lr = 1e-5
    convs_max_lr = 1e-8
    blend_max_lr = 1e-4
    norm_max_lr = 1e-7
    weight_decay = 0.01

    param_groups = [
        # Encoder
        {"params": enc_decay, "lr": enc_max_lr, "weight_decay": weight_decay},
        {"params": enc_no_decay, "lr": enc_max_lr, "weight_decay": 0.0},
        # Linear heads (lin.car, lin.spot)
        {"params": head_decay, "lr": head_max_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": head_max_lr, "weight_decay": 0.0},
        # LogitBlendHead (alpha, bias) — usually lower LR for stability
        {"params": blend_decay, "lr": blend_max_lr, "weight_decay": weight_decay},
        {"params": blend_no_decay, "lr": blend_max_lr, "weight_decay": 0.0},
        # GNN conv layers
        {"params": convs_decay, "lr": convs_max_lr, "weight_decay": weight_decay},
        {"params": convs_no_decay, "lr": convs_max_lr, "weight_decay": 0.0},
        # Norm layers (LayerNorm)
        {"params": norm_decay, "lr": norm_max_lr, "weight_decay": weight_decay},
        {"params": norm_no_decay, "lr": norm_max_lr, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, betas=(0.9, 0.999))
    total_epochs = 100

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[
            enc_max_lr,
            enc_max_lr,  # encoder (decay, no_decay)
            head_max_lr,
            head_max_lr,  # lin heads
            blend_max_lr,
            blend_max_lr,  # logit_blend
            convs_max_lr,
            convs_max_lr,  # GNN convs
            norm_max_lr,
            norm_max_lr,  # norm layers
        ],
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
        pct_start=0.3,  # warm-up portion
        anneal_strategy="cos",
        div_factor=25.0,  # initial_lr = max_lr / div_factor
        final_div_factor=1e3,  # final_lr = initial_lr / final_div_factor
    )

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch["spot"].score)[
                "spot"
            ]
            probs = torch.sigmoid(out)
            loss = margin_ranking_loss(probs, batch["spot"].y.float(), margin=1)

            optimizer.zero_grad(set_to_none=True)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                total_loss += [loss.item()]

                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=sum(total_loss) / len(total_loss), lr=current_lr
                    )
                else:
                    pbar.set_postfix(loss=sum(total_loss) / len(total_loss))

            if scheduler is not None:
                scheduler.step()

        # if epoch % 100 == 0:
        #     evaluate(model, graph_dir, test_loader, device, labelled_spots_dir, val_subsets)

    os.makedirs(weight_root, exist_ok=True)
    weight_file = os.path.join(weight_root, "final.pt")
    torch.save(model.state_dict(), weight_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GraphSpot GNN on MetaPKLot dataset"
    )
    parser.add_argument(
        "--root", help="Path to the annotation files folder", required=True
    )
    parser.add_argument("--model", default="mask")
    parser.add_argument("--exp", help="Experiment name", required=True)
    parser.add_argument(
        "--output_dir", help="Path to the output directory", required=True
    )

    args = parser.parse_args()
    main(args)
