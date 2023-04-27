import cv2

series_name_array = ["car", "cup"]
series_frame_count_array = [374, 303]

for k in range(2):
    series_name = series_name_array[k]
    series_frame_count = series_frame_count_array[k]
    for i in range(1, series_frame_count + 1):
        frame = cv2.imread(f"origin/{series_name}_frames/{i:08}.jpg")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with open(f"frames-graybin/{series_name}/{i}.bin", "wb") as f:
            f.write(bytes(frame_gray))
