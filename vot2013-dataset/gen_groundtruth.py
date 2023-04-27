import csv

for series_name in ["car", "cup"]:
    groundtruth = []
    with open(f"origin/{series_name}/groundtruth.txt", "r") as f:
        for row in csv.reader(f):
            groundtruth.append([float(i) for i in row])
    groundtruth_center = [[x + w / 2, y + h / 2] for [x, y, w, h] in groundtruth]
    with open(f"groundtruth/{series_name}.py", "w") as f:
        f.write(f"# [xc, yc]\ngroundtruth_center = [\n")
        f.write(
            ", ".join(
                [f"[{center[0]:.1f}, {center[1]:.1f}]" for center in groundtruth_center]
            )
        )
        f.write("\n]\n")
