import cv2

series_name = "car"  # car, cup

if series_name == "car":
    frame_count = 374
    frame_size = [320, 240]
    frame_rate = 30
    x, y, w, h = [257.0, 163.0, 57.0, 36.0]
elif series_name == "cup":
    frame_count = 303
    frame_size = [320, 240]
    frame_rate = 30
    x, y, w, h = [124.67, 92.308, 46.73, 58.572]

xc: float = x + w / 2
yc: float = y + h / 2
result: list[list[int]] = [[int(xc), int(yc)]]

# ----------------

from mosse import Mosse

mosse = Mosse()

writer = cv2.VideoWriter(
    f"rgbresult/{series_name}.mp4",
    cv2.VideoWriter_fourcc("H", "2", "6", "4"),
    frame_rate,
    frame_size,
)

# [init]

with open(f"frames-graybin/{series_name}/1.bin", "rb") as f:
    frame_init = f.read()
mosse.init()

# [update]

for i in range(2, frame_count + 1):
    with open(f"frames-graybin/{series_name}/{i}.bin", "rb") as f:
        frame = f.read()
    xc, yc = mosse.update()
    print(f"{i}\t({xc=}, {yc=})")
    result.append([int(xc), int(yc)])

# [quit]

print(f"{result=}")
with open(f"result-{series_name}.py", "w") as f:
    f.write(f"result = {result}\n")