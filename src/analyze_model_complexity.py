import os
import sys
import torch
import timm
import pandas as pd

from pathlib import Path


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)


# ============================================================
# IMPORT PROPOSED MODEL
# ============================================================

from proposed_attention_efficientnet import (
    AttentionEfficientNetB0
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_CLASSES = 6


# ============================================================
# COUNT PARAMETERS
# ============================================================

def count_parameters(model):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters
    )


# ============================================================
# CALCULATE MODEL SIZE
# ============================================================

def calculate_model_size_mb(model):

    parameter_size = 0

    for parameter in model.parameters():

        parameter_size += (
            parameter.nelement()
            * parameter.element_size()
        )

    buffer_size = 0

    for buffer in model.buffers():

        buffer_size += (
            buffer.nelement()
            * buffer.element_size()
        )

    total_size_bytes = (
        parameter_size
        + buffer_size
    )

    total_size_mb = (
        total_size_bytes
        / (1024 ** 2)
    )

    return total_size_mb


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {}


    # --------------------------------------------------------
    # CONVNEXTV2-TINY
    # --------------------------------------------------------

    models[
        "ConvNeXtV2-Tiny"
    ] = timm.create_model(

        "convnextv2_tiny",

        pretrained=False,

        num_classes=NUM_CLASSES
    )


    # --------------------------------------------------------
    # DENSENET121
    # --------------------------------------------------------

    models[
        "DenseNet121"
    ] = timm.create_model(

        "densenet121",

        pretrained=False,

        num_classes=NUM_CLASSES
    )


    # --------------------------------------------------------
    # EFFICIENTNET-B0
    # --------------------------------------------------------

    models[
        "EfficientNet-B0"
    ] = timm.create_model(

        "efficientnet_b0",

        pretrained=False,

        num_classes=NUM_CLASSES
    )


    # --------------------------------------------------------
    # PROPOSED MODEL
    # --------------------------------------------------------

    models[
        "Proposed Attention-EfficientNet-B0"
    ] = AttentionEfficientNetB0(

        num_classes=NUM_CLASSES,

        pretrained=False
    )


    return models


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "MODEL COMPLEXITY ANALYSIS"
    )

    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )


    # --------------------------------------------------------
    # CREATE MODELS
    # --------------------------------------------------------

    print(
        "\nCreating models..."
    )

    models = create_models()


    # --------------------------------------------------------
    # ANALYZE MODELS
    # --------------------------------------------------------

    results = []


    for model_name, model in models.items():

        print(
            f"\nAnalyzing: {model_name}"
        )


        # ----------------------------------------------------
        # MOVE MODEL TO DEVICE
        # ----------------------------------------------------

        model = model.to(
            DEVICE
        )


        # ----------------------------------------------------
        # PARAMETER COUNT
        # ----------------------------------------------------

        (
            total_parameters,
            trainable_parameters
        ) = count_parameters(
            model
        )


        # ----------------------------------------------------
        # MODEL SIZE
        # ----------------------------------------------------

        model_size_mb = (
            calculate_model_size_mb(
                model
            )
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        results.append({

            "Model":
                model_name,

            "Total_Parameters":
                total_parameters,

            "Trainable_Parameters":
                trainable_parameters,

            "Total_Parameters_Millions":
                total_parameters / 1_000_000,

            "Model_Size_MB":
                model_size_mb

        })


        print(
            f"Total Parameters: "
            f"{total_parameters:,}"
        )

        print(
            f"Trainable Parameters: "
            f"{trainable_parameters:,}"
        )

        print(
            f"Model Size: "
            f"{model_size_mb:.2f} MB"
        )


        # ----------------------------------------------------
        # CLEAN GPU MEMORY
        # ----------------------------------------------------

        del model

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    results_dataframe = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    metrics_directory = (
        PROJECT_ROOT
        / "results"
        / "metrics"
    )

    metrics_directory.mkdir(

        parents=True,

        exist_ok=True
    )


    output_path = (

        metrics_directory

        / "model_complexity_comparison.csv"

    )


    results_dataframe.to_csv(

        output_path,

        index=False

    )


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPLEXITY COMPARISON"
    )

    print(
        "=" * 70
    )


    print(

        results_dataframe.to_string(

            index=False

        )

    )


    print(
        "\nSaved to:"
    )

    print(
        output_path
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
