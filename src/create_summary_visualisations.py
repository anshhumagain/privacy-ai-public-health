from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = BASE_DIR / "graphs"

SUMMARY_SETS = {
    "dp": {
        "title": "Differential Privacy — Summary Visualisations",
        "files": [
            "dp_covid_metrics.png",
            "dp_covid_privacy_utility.png",
            "dp_covid_runtime.png",
            "dp_nhanes_metrics.png",
            "dp_nhanes_privacy_utility.png",
            "dp_nhanes_runtime.png",
        ],
        "output": "summary_dp.png",
    },
    "fl": {
        "title": "Federated Learning — Summary Visualisations",
        "files": [
            "fl_covid_metrics.png",
            "fl_covid_privacy_utility.png",
            "fl_covid_runtime.png",
            "fl_nhanes_metrics.png",
            "fl_nhanes_privacy_utility.png",
            "fl_nhanes_runtime.png",
        ],
        "output": "summary_fl.png",
    },
    "he": {
        "title": "Homomorphic Encryption — Summary Visualisations",
        "files": [
            "he_covid_metrics.png",
            "he_covid_privacy_utility.png",
            "he_covid_runtime.png",
            "he_nhanes_metrics.png",
            "he_nhanes_privacy_utility.png",
            "he_nhanes_runtime.png",
        ],
        "output": "summary_he.png",
    },
    "kanon": {
        "title": "K-Anonymity — Summary Visualisations",
        "files": [
            "kanon_covid_metrics.png",
            "kanon_covid_privacy_utility.png",
            "kanon_covid_runtime.png",
            "kanon_nhanes_metrics.png",
            "kanon_nhanes_privacy_utility.png",
            "kanon_nhanes_runtime.png",
        ],
        "output": "summary_kanon.png",
    },
}

def resize_to_width(img, width):
    ratio = width / img.width
    height = int(img.height * ratio)
    return img.resize((width, height), Image.LANCZOS)

def make_summary(title, files, output):
    images = []

    for file in files:
        path = GRAPHS_DIR / file
        if not path.exists():
            raise FileNotFoundError(f"Missing graph: {path}")
        img = Image.open(path).convert("RGB")
        images.append(img)

    cell_width = 900
    padding = 40
    title_height = 90
    label_height = 35

    resized = [resize_to_width(img, cell_width) for img in images]
    cell_height = max(img.height for img in resized)

    canvas_width = padding * 3 + cell_width * 2
    canvas_height = title_height + (cell_height + label_height + padding) * 3 + padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("Arial Bold.ttf", 34)
        label_font = ImageFont.truetype("Arial Bold.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.text((padding, 25), title, fill="black", font=title_font)

    labels = [
        "COVID-19 Metrics",
        "COVID-19 Privacy-Utility",
        "COVID-19 Runtime",
        "NHANES Metrics",
        "NHANES Privacy-Utility",
        "NHANES Runtime",
    ]

    positions = [
        (padding, title_height),
        (padding * 2 + cell_width, title_height),
        (padding, title_height + cell_height + label_height + padding),
        (padding * 2 + cell_width, title_height + cell_height + label_height + padding),
        (padding, title_height + 2 * (cell_height + label_height + padding)),
        (padding * 2 + cell_width, title_height + 2 * (cell_height + label_height + padding)),
    ]

    # Reorder to make rows read better:
    # Row 1 = metrics, Row 2 = privacy utility, Row 3 = runtime
    order = [0, 3, 1, 4, 2, 5]
    ordered_images = [resized[i] for i in order]
    ordered_labels = [labels[i] for i in order]

    for img, label, pos in zip(ordered_images, ordered_labels, positions):
        x, y = pos
        draw.text((x, y), label, fill="black", font=label_font)
        canvas.paste(img, (x, y + label_height))

    output_path = GRAPHS_DIR / output
    canvas.save(output_path, quality=95)
    print(f"Saved: {output_path}")

for item in SUMMARY_SETS.values():
    make_summary(item["title"], item["files"], item["output"])

print("\nAll summary visualisations saved.")
