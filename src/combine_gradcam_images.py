from pathlib import Path
from PIL import Image, ImageDraw


INPUT_DIR = Path("results/figures/gradcam_selected_cases")

OUTPUT_PATH = (
    INPUT_DIR /
    "combined_gradcam_analysis.png"
)


files = sorted(
    [
        file
        for file in INPUT_DIR.glob("*.png")
        if file.name != "combined_gradcam_analysis.png"
    ]
)


images = []

for file in files:

    image = Image.open(file).convert("RGB")

    images.append(
        (file.name, image)
    )


max_width = max(
    image.width
    for _, image in images
)


label_height = 50


total_height = sum(
    image.height + label_height
    for _, image in images
)


canvas = Image.new(
    "RGB",
    (max_width, total_height),
    "white"
)


draw = ImageDraw.Draw(canvas)


y_position = 0


for filename, image in images:

    draw.text(
        (20, y_position + 15),
        filename[:100],
        fill="black"
    )

    y_position += label_height

    x_position = (
        max_width - image.width
    ) // 2

    canvas.paste(
        image,
        (x_position, y_position)
    )

    y_position += image.height


canvas.save(
    OUTPUT_PATH,
    quality=95
)


print(
    "\n"
    + "=" * 60
)

print(
    "COMBINED GRAD-CAM IMAGE CREATED"
)

print(
    "=" * 60
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)