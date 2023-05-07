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


from result import result

writer = cv2.VideoWriter(
    f"result/{series_name}.mp4",
    cv2.VideoWriter_fourcc("H", "2", "6", "4"),
    frame_rate,
    frame_size,
)

for i in range(1, frame_count + 1):
    frame = cv2.imread(f"../vot2013-dataset/origin/{series_name}_frames/{i:08}.jpg")
    xc, yc = result[i - 1]
    xc, yc = int(xc), int(yc)
    frame_rectangle = cv2.rectangle(
        img=frame,
        pt1=(xc - 16, yc - 16),
        pt2=(xc + 16, yc + 16),
        color=(0, 0, 255),
        thickness=1,
    )
    # cv2.imwrite(f"result/{series_name}/{i}.png", frame_rectangle)
    writer.write(frame_rectangle)
