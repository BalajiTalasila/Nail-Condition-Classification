import os
import sys
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_DIR)


from src.proposed_attention_efficientnet import (
    AttentionEfficientNetB0
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "train"
)

VAL_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "val"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "models"
)

METRICS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "metrics"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


IMAGE_SIZE = 224
NUM_CLASSES = 6

BATCH_SIZE = 16

NUM_EPOCHS = 80

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.00001

DROPOUT = 0.30

PATIENCE = 15

SEED = 42


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATA TRANSFORMATIONS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        15
    ),

    transforms.ColorJitter(

        brightness=0.15,

        contrast=0.15,

        saturation=0.10

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224
            ,
            0.225
        ]
    )
])


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():


    print("\n" + "=" * 70)

    print(
        "TRAINING PROPOSED ATTENTION-EFFICIENTNET-B0 MODEL"
    )

    print("=" * 70)


    print(
        f"\nDevice: {DEVICE}"
    )


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print(
        "\nLoading datasets..."
    )


    train_dataset = datasets.ImageFolder(

        TRAIN_DIR,

        transform=train_transform
    )


    val_dataset = datasets.ImageFolder(

        VAL_DIR,

        transform=validation_transform
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True
    )


    print(

        f"Training images: "

        f"{len(train_dataset)}"

    )


    print(

        f"Validation images: "

        f"{len(val_dataset)}"

    )


    print(

        f"Classes: "

        f"{train_dataset.classes}"

    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print(
        "\nCreating proposed model..."
    )


    model = AttentionEfficientNetB0(

        num_classes=NUM_CLASSES,

        dropout=DROPOUT,

        pretrained=True
    )


    model = model.to(

        DEVICE
    )


    print(

        "? Proposed Attention-EfficientNet-B0 created"
    )


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=5
    )


    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    best_validation_accuracy = 0.0

    best_validation_loss = float("inf")

    patience_counter = 0


    history = []


    model_path = os.path.join(

        MODEL_DIR,

        "proposed_attention_efficientnet_b0_best.pth"
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(NUM_EPOCHS):


        print(

            "\n" + "-" * 70
        )


        print(

            f"Epoch "

            f"{epoch + 1}/{NUM_EPOCHS}"
        )


        print(

            "-" * 70
        )


        epoch_start_time = time.time()


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()


        running_loss = 0.0

        correct_predictions = 0

        total_predictions = 0


        progress_bar = tqdm(

            train_loader,

            desc="Training"
        )


        for images, labels in progress_bar:


            images = images.to(

                DEVICE
            )


            labels = labels.to(

                DEVICE
            )


            optimizer.zero_grad()


            outputs = model(

                images
            )


            loss = criterion(

                outputs,

                labels
            )


            loss.backward()


            optimizer.step()


            running_loss += (

                loss.item()

                * images.size(0)
            )


            _, predicted = torch.max(

                outputs,

                1
            )


            correct_predictions += (

                predicted == labels

            ).sum().item()


            total_predictions += (

                labels.size(0)
            )


            progress_bar.set_postfix(

                loss=f"{loss.item():.4f}"
            )


        train_loss = (

            running_loss

            / len(train_dataset)
        )


        train_accuracy = (

            100

            * correct_predictions

            / total_predictions
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        validation_loss = 0.0

        validation_correct = 0

        validation_total = 0


        with torch.no_grad():


            for images, labels in val_loader:


                images = images.to(

                    DEVICE
                )


                labels = labels.to(

                    DEVICE
                )


                outputs = model(

                    images
                )


                loss = criterion(

                    outputs,

                    labels
                )


                validation_loss += (

                    loss.item()

                    * images.size(0)
                )


                _, predicted = torch.max(

                    outputs,

                    1
                )


                validation_correct += (

                    predicted == labels

                ).sum().item()


                validation_total += (

                    labels.size(0)
                )


        validation_loss = (

            validation_loss

            / len(val_dataset)
        )


        validation_accuracy = (

            100

            * validation_correct

            / validation_total
        )


        scheduler.step(

            validation_loss
        )


        current_learning_rate = (

            optimizer.param_groups[0]["lr"]
        )


        epoch_time = (

            time.time()

            - epoch_start_time
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        print(

            f"\nTraining Loss: "

            f"{train_loss:.4f}"
        )


        print(

            f"Training Accuracy: "

            f"{train_accuracy:.2f}%"
        )


        print(

            f"Validation Loss: "

            f"{validation_loss:.4f}"
        )


        print(

            f"Validation Accuracy: "

            f"{validation_accuracy:.2f}%"
        )


        print(

            f"Learning Rate: "

            f"{current_learning_rate:.8f}"
        )


        print(

            f"Epoch Time: "

            f"{epoch_time:.2f} seconds"
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history.append({

            "Epoch":

                epoch + 1,

            "Train_Loss":

                train_loss,

            "Train_Accuracy":

                train_accuracy,

            "Validation_Loss":

                validation_loss,

            "Validation_Accuracy":

                validation_accuracy,

            "Learning_Rate":

                current_learning_rate,

            "Epoch_Time_Seconds":

                epoch_time
        })


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_accuracy > best_validation_accuracy:


            best_validation_accuracy = (

                validation_accuracy
            )


            best_validation_loss = (

                validation_loss
            )


            patience_counter = 0


            checkpoint = {

                "epoch":

                    epoch + 1,

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

                    "proposed_attention_efficientnet_b0",

                "image_size":

                    IMAGE_SIZE,

                "random_seed":

                    SEED
            }


            torch.save(

                checkpoint,

                model_path
            )


            print(

                "\n? BEST MODEL SAVED"
            )


            print(

                f"Best Validation Accuracy: "

                f"{best_validation_accuracy:.2f}%"
            )


        else:


            patience_counter += 1


            print(

                f"\nEarly stopping counter: "

                f"{patience_counter}/{PATIENCE}"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= PATIENCE:


            print(

                "\nEarly stopping activated."
            )


            break


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_dataframe = pd.DataFrame(

        history
    )


    history_path = os.path.join(

        METRICS_DIR,

        "proposed_attention_efficientnet_b0_training_history.csv"
    )


    history_dataframe.to_csv(

        history_path,

        index=False
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(

        "\n" + "=" * 70
    )


    print(

        "PROPOSED MODEL TRAINING COMPLETED"
    )


    print(

        "=" * 70
    )


    print(

        f"\nBest Validation Accuracy: "

        f"{best_validation_accuracy:.2f}%"
    )


    print(

        f"Best Validation Loss: "

        f"{best_validation_loss:.4f}"
    )


    print(

        f"\nModel saved to:"

    )


    print(

        model_path
    )


    print(

        "\nTraining history saved to:"

    )


    print(

        history_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()
