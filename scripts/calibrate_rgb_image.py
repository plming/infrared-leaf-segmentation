"""
RGB 이미지와 IR 이미지의 촬영 위치 차이를 보정하기 위한 스크립트
"""
import cv2

config = {
    "path": "../ginseng-rgb/camimage_20220512_0800.jpg",
    "zoom_ratio": 1.3,
    "offset_x": 2,
    "offset_y": 7,
    "width": 160,
    "height": 120
}

if __name__ == "__main__":
    image = cv2.imread(config["path"], cv2.IMREAD_COLOR)
    image = cv2.resize(image,
                       dsize=(config["width"], config["height"]),
                       interpolation=cv2.INTER_AREA)

    image = cv2.resize(image,
                       dsize=None,
                       fx=config["zoom_ratio"],
                       fy=config["zoom_ratio"],
                       interpolation=cv2.INTER_LANCZOS4)

    y, x, _ = image.shape
    start_x = x // 2 - (config["width"] // 2) + config["offset_x"]
    start_y = y // 2 - (config["height"] // 2) + config["offset_y"]

    image = image[start_y:start_y + config["height"], start_x:start_x + config["width"]]

    new_path = config["path"].replace(".jpg", "_calibrated.jpg")
    cv2.imwrite(new_path, image)
    print(f"saved to {new_path}")
