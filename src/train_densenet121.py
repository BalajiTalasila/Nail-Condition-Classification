# ============================================================
# DENSENET121 TRAINING
# SIX-CLASS NAIL DISEASE CLASSIFICATION
# ============================================================

import os
import random
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MODEL_NAME = "densenet121"

NUM_CLASSES = 6

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 10


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

MODELS_DIR = (
    RESULTS_DIR
    / "models"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_data_loaders():

    train_dataset = datasets.ImageFolder(

        DATA_DIR / "train",

        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(

        DATA_DIR / "val",

        transform=validation_transform
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True
    )

    return (

        train_dataset,

        validation_dataset,

        train_loader,

        validation_loader
    )


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_dataset):

    targets = np.array(
        train_dataset.targets
    )

    class_counts = np.bincount(
        targets
    )

    total_samples = len(
        targets
    )

    class_weights = (

        total_samples

        / (

            NUM_CLASSES
            * class_counts
        )
    )

    weights_tensor = torch.tensor(

        class_weights,

        dtype=torch.float32

    ).to(DEVICE)

    return (

        weights_tensor,

        class_counts
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(

    model,

    data_loader,

    criterion,

    optimizer,

    scaler

):

    model.train()

    running_loss = 0.0

    correct_predictions = 0

    total_predictions = 0

    progress_bar = tqdm(

        data_loader,

        desc="Training",

        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(

            DEVICE,

            non_blocking=True
        )

        labels = labels.to(

            DEVICE,

            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(

            device_type=DEVICE.type,

            enabled=(
                DEVICE.type == "cuda"
            )
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (

            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(

            outputs,

            dim=1
        )

        correct_predictions += (

            predictions == labels

        ).sum().item()

        total_predictions += (

            labels.size(0)
        )

        progress_bar.set_postfix({

            "loss":
            f"{loss.item():.4f}"

        })

    epoch_loss = (

        running_loss
        / total_predictions
    )

    epoch_accuracy = (

        correct_predictions
        / total_predictions
        * 100
    )

    return (

        epoch_loss,

        epoch_accuracy
    )


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate(

    model,

    data_loader,

    criterion

):

    model.eval()

    running_loss = 0.0

    correct_predictions = 0

    total_predictions = 0

    with torch.no_grad():

        progress_bar = tqdm(

            data_loader,

            desc="Validation",

            leave=False
        )

        for images, labels in progress_bar:

            images = images.to(

                DEVICE,

                non_blocking=True
            )

            labels = labels.to(

                DEVICE,

                non_blocking=True
            )

            with torch.amp.autocast(

                device_type=DEVICE.type,

                enabled=(
                    DEVICE.type == "cuda"
                )
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (

                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(

                outputs,

                dim=1
            )

            correct_predictions += (

                predictions == labels

            ).sum().item()

            total_predictions += (

                labels.size(0)
            )

    validation_loss = (

        running_loss
        / total_predictions
    )

    validation_accuracy = (

        correct_predictions
        / total_predictions
        * 100
    )

    return (

        validation_loss,

        validation_accuracy
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
parser = argparse.ArgumentParser(
    description="Train DENSENET121"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

args = parser.parse_args()

SEED = args.seed

set_seed(SEED)

print(f"\nRandom Seed: {SEED}")

def main():

    print("\n" + "=" * 70)

    print(
        "DENSENET121 TRAINING"
    )

    print("=" * 70)

    set_seed(
        RANDOM_SEED
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(

            "GPU: "
            + torch.cuda.get_device_name(0)

        )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (

        train_dataset,

        validation_dataset,

        train_loader,

        validation_loader

    ) = create_data_loaders()

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        "\nClass mapping:"
    )

    for class_name, index in (
        train_dataset.class_to_idx.items()
    ):

        print(
            f"{index}: "
            f"{class_name}"
        )


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights, class_counts = (

        calculate_class_weights(
            train_dataset
        )
    )

    print(
        "\nClass distribution:"
    )

    for index, count in enumerate(
        class_counts
    ):

        print(
            f"{train_dataset.classes[index]}: "
            f"{count}"
        )

    print(
        "\nClass weights:"
    )

    for index, weight in enumerate(
        class_weights.cpu().numpy()
    ):

        print(
            f"{train_dataset.classes[index]}: "
            f"{weight:.4f}"
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading pretrained DENSENET121..."
    )

    model = timm.create_model(

        MODEL_NAME,

        pretrained=True,

        num_classes=NUM_CLASSES
    )

    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=0.05
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3
    )


    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(

        enabled=(
            DEVICE.type == "cuda"
        )
    )


    # --------------------------------------------------------
    # TRAINING VARIABLES
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    patience_counter = 0

    training_history = []

    start_time = time.time()


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    print(
        "\nStarting training...\n"
    )

    for epoch in range(

        1,

        NUM_EPOCHS + 1

    ):

        epoch_start_time = time.time()

        print(
            f"\nEpoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = (

            train_one_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scaler
            )
        )

        validation_loss, validation_accuracy = (

            validate(

                model,

                validation_loader,

                criterion
            )
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (

            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (

            time.time()
            - epoch_start_time
        )

        print(

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(

            f"Validation Loss: "
            f"{validation_loss:.4f} | "

            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

        print(

            f"Learning Rate: "
            f"{current_lr:.8f}"
        )

        print(

            f"Epoch Time: "
            f"{epoch_time:.2f} seconds"
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        training_history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "validation_loss": validation_loss,

            "validation_accuracy": validation_accuracy,

            "learning_rate": current_lr,

            "epoch_time_seconds": epoch_time
        })


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_validation_accuracy = (
                validation_accuracy
            )

            patience_counter = 0

            checkpoint = {

                "epoch": epoch,

                "model_state_dict":
                model.state_dict(),

                "optimizer_state_dict":
                optimizer.state_dict(),

                "validation_loss":
                validation_loss,

                "validation_accuracy":
                validation_accuracy,

                "class_names":
                train_dataset.classes,

                "model_name":
                MODEL_NAME,

                "image_size":
                IMAGE_SIZE,

                "random_seed":
                RANDOM_SEED
            }

            checkpoint_path = (

                MODELS_DIR

                / "densenet121_best.pth"
            )

            torch.save(

                checkpoint,

                checkpoint_path
            )

            print(
                "\n✓ Best model saved!"
            )

        else:

            patience_counter += 1

            print(

                f"\nNo validation loss improvement. "

                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(

        training_history
    )

    history_path = (

        METRICS_DIR

        / "densenet121_training_history.csv"
    )

    history_df.to_csv(

        history_path,

        index=False
    )


    # --------------------------------------------------------
    # TRAINING SUMMARY
    # --------------------------------------------------------

    total_training_time = (

        time.time()
        - start_time
    )

    print("\n" + "=" * 70)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 70)

    print(

        f"\nBest Validation Loss: "
        f"{best_validation_loss:.4f}"
    )

    print(

        f"Validation Accuracy at "
        f"Best Loss: "
        f"{best_validation_accuracy:.2f}%"
    )

    print(

        f"Total Training Time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(

        "\nBest model saved at:"
    )

    print(

        MODELS_DIR
        / "densenet121_best.pth"
    )

    print(

        "\nTraining history saved at:"
    )

    print(
        history_path
    )


if __name__ == "__main__":

    main()
