from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "evaluation" / "unseen_source_test"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 80)
print("UNSEEN-SOURCE DATASET MODEL EVALUATION")
print("=" * 80)

print(f"\nDevice: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


class_names = dataset.classes
num_classes = len(class_names)


print(f"\nTotal evaluation images: {len(dataset)}")

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


MODEL_CONFIGS = {
    "convnextv2_tiny": {
        "checkpoint": "convnextv2_tiny_best.pth",
        "architecture": "convnextv2_tiny"
    },
    "densenet121": {
        "checkpoint": "densenet121_best.pth",
        "architecture": "densenet121"
    },
    "efficientnet_b0": {
        "checkpoint": "efficientnet_b0_best.pth",
        "architecture": "efficientnet_b0"
    },
    "proposed_attention_efficientnet_b0": {
        "checkpoint": "proposed_attention_efficientnet_b0_best.pth",
        "architecture": "proposed_attention_efficientnet_b0"
    }
}


from proposed_attention_efficientnet import AttentionEfficientNetB0



def create_model(model_name, num_classes):

    if model_name == "densenet121":

        model = models.densenet121(
            weights=None
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

        return model


    elif model_name == "efficientnet_b0":

        import timm

        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes
        )

        return model


    elif model_name == "convnextv2_tiny":

        import timm

        model = timm.create_model(
            "convnextv2_tiny",
            pretrained=False,
            num_classes=num_classes
        )

        return model


    elif model_name == "proposed_attention_efficientnet_b0":

        model = AttentionEfficientNetB0(
            num_classes=num_classes
        )

        return model


    else:

        raise ValueError(
            f"Unknown architecture: {model_name}"
        )


def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    new_state_dict = {}


    for key, value in state_dict.items():

        new_key = key.replace(
            "module.",
            ""
        )

        new_state_dict[new_key] = value


    model.load_state_dict(
        new_state_dict,
        strict=True
    )


    return model


results = []


for model_name, config in MODEL_CONFIGS.items():

    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)


    checkpoint_path = (
        MODELS_DIR /
        config["checkpoint"]
    )


    if not checkpoint_path.exists():

        print("\nWARNING: Model checkpoint not found:")
        print(checkpoint_path)

        continue


    print("\nCheckpoint found:")
    print(checkpoint_path)


    try:

        model = create_model(
            config["architecture"],
            num_classes
        )


        model = load_checkpoint(
            model,
            checkpoint_path
        )


        model = model.to(
            DEVICE
        )


        model.eval()


        all_true = []
        all_predicted = []
        all_image_paths = []
        all_confidences = []
        all_probabilities = []


        image_index = 0


        with torch.no_grad():

            for images, labels in dataloader:

                batch_size = images.size(
                    0
                )


                images = images.to(
                    DEVICE
                )


                labels = labels.to(
                    DEVICE
                )


                outputs = model(
                    images
                )


                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )


                confidences, predictions = torch.max(
                    probabilities,
                    dim=1
                )


                all_confidences.extend(
                    confidences.cpu().numpy()
                )


                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )


                batch_indices = range(
                    image_index,
                    image_index + batch_size
                )


                for dataset_index in batch_indices:

                    image_path, _ = dataset.samples[
                        dataset_index
                    ]


                    all_image_paths.append(
                        str(image_path)
                    )


                image_index += batch_size


                all_true.extend(
                    labels.cpu().numpy()
                )


                all_predicted.extend(
                    predictions.cpu().numpy()
                )


        accuracy = accuracy_score(
            all_true,
            all_predicted
        )


        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                all_true,
                all_predicted,
                average="weighted",
                zero_division=0
            )
        )


        print(
            f"\nAccuracy: {accuracy * 100:.2f}%"
        )

        print(
            f"Precision: {precision * 100:.2f}%"
        )

        print(
            f"Recall: {recall * 100:.2f}%"
        )

        print(
            f"F1 Score: {f1 * 100:.2f}%"
        )


        # ============================================================
        # DETAILED PER-CLASS EVALUATION
        # ============================================================

        print("\n" + "-" * 80)
        print("PER-CLASS CLASSIFICATION REPORT")
        print("-" * 80)

        report = classification_report(
            all_true,
            all_predicted,
            target_names=class_names,
            digits=4,
            zero_division=0
        )

        print(report)


        # ============================================================
        # CONFUSION MATRIX
        # ============================================================

        cm = confusion_matrix(
            all_true,
            all_predicted
        )


        print("-" * 80)
        print("CONFUSION MATRIX")
        print("-" * 80)

        cm_df = pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names
        )

        print(cm_df.to_string())


        # ============================================================
        # SAVE DETAILED MODEL RESULTS
        # ============================================================

        detailed_output_dir = (
            OUTPUT_DIR /
            "unseen_source_detailed_results"
        )

        detailed_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        cm_file = (
            detailed_output_dir /
            f"{model_name}_confusion_matrix.csv"
        )

        cm_df.to_csv(
            cm_file
        )


        report_dict = classification_report(
            all_true,
            all_predicted,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )


        report_df = pd.DataFrame(
            report_dict
        ).transpose()


        report_file = (
            detailed_output_dir /
            f"{model_name}_classification_report.csv"
        )

        report_df.to_csv(
            report_file
        )


        print("\nDetailed files saved:")

        print(cm_file)

        print(report_file)


        # ============================================================
        # SAVE IMAGE-LEVEL PREDICTIONS
        # ============================================================

        predictions_df = pd.DataFrame({

            "image_path": all_image_paths,

            "true_class_index": all_true,

            "true_class": [
                class_names[index]
                for index in all_true
            ],

            "predicted_class_index": all_predicted,

            "predicted_class": [
                class_names[index]
                for index in all_predicted
            ],

            "confidence": all_confidences

        })


        probability_df = pd.DataFrame(
            all_probabilities,
            columns=[
                f"probability_{class_name}"
                for class_name in class_names
            ]
        )


        predictions_df = pd.concat(

            [
                predictions_df.reset_index(
                    drop=True
                ),

                probability_df.reset_index(
                    drop=True
                )
            ],

            axis=1

        )


        predictions_file = (
            detailed_output_dir /
            f"{model_name}_image_predictions.csv"
        )


        predictions_df.to_csv(
            predictions_file,
            index=False
        )


        print(predictions_file)


        results.append({

            "model": model_name,

            "total_images": len(dataset),

            "accuracy_percent": accuracy * 100,

            "precision_percent": precision * 100,

            "recall_percent": recall * 100,

            "f1_score_percent": f1 * 100

        })


        del model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    except Exception as error:

        print(
            f"\nERROR while evaluating {model_name}:"
        )

        print(error)


print("\n" + "=" * 80)
print("FINAL UNSEEN-SOURCE MODEL COMPARISON")
print("=" * 80)


results_df = pd.DataFrame(
    results
)


if results_df.empty:

    print(
        "\nERROR: No models were successfully evaluated."
    )

    raise SystemExit(1)


results_df = results_df.sort_values(
    "accuracy_percent",
    ascending=False
)


print("\n")

print(
    results_df.to_string(
        index=False
    )
)


summary_file = (
    OUTPUT_DIR /
    "unseen_source_model_comparison.csv"
)


results_df.to_csv(
    summary_file,
    index=False
)


print("\nSummary file saved:")

print(
    summary_file
)


print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)
