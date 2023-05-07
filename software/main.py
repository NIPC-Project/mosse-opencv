import cv2
import numpy as np

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
    f"result/{series_name}.mp4",
    cv2.VideoWriter_fourcc("H", "2", "6", "4"),
    frame_rate,
    frame_size,
)

# [init]

with open(f"../vot2013-dataset/frames-graybin/{series_name}/1.bin", "rb") as f:
    frame_init = f.read()
mosse.init(frame=np.array(list(frame_init)).reshape((240, 320)), xc0=xc, yc0=yc)

# [update]

for i in range(2, frame_count + 1):
    with open(f"../vot2013-dataset/frames-graybin/{series_name}/{i}.bin", "rb") as f:
        frame = f.read()
    xc, yc = mosse.update(frame=np.array(list(frame)).reshape((240, 320)))
    print(f"\n[{i}]\t({xc=}, {yc=})")
    result.append([int(xc), int(yc)])

    frame = cv2.imread(f"../vot2013-dataset/origin/{series_name}_frames/{i:08}.jpg")
    frame_rectangle = cv2.rectangle(
        img=frame,
        pt1=(int(xc) - 16, int(yc) - 16),
        pt2=(int(xc) + 16, int(yc) + 16),
        color=(0, 0, 255),
        thickness=1,
    )
    writer.write(frame_rectangle)

# [quit]

print(f"{result=}")
with open(f"result/{series_name}.py", "w") as f:
    f.write(f"result = {result}\n")
