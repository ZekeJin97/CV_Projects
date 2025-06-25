import os
import json

def convert_labelme_folder_to_csv(input_folder):
    output_folder = input_folder.rstrip("/\\") + "_csv"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            csv_name = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(output_folder, csv_name)

            with open(json_path, 'r') as f:
                data = json.load(f)

            shapes = data.get('shapes', [])
            with open(csv_path, 'w') as out:
                out.write(f"{len(shapes)}\n")
                for shape in shapes:
                    label = shape.get("label", "unknown")
                    points = shape.get("points", [])
                    if len(points) < 2:
                        continue  # skip broken annotations

                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    minX, minY = min(x_coords), min(y_coords)
                    maxX, maxY = max(x_coords), max(y_coords)
                    out.write(f"{int(minX)} {int(minY)} {int(maxX)} {int(maxY)} {label}\n")

    print(f"✅ All JSONs converted to CSV in: {output_folder}")

# USAGE
convert_labelme_folder_to_csv("../Data")

