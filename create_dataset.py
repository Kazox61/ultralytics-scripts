import shutil
import os
import random
import cv2
from pathlib import Path
import argparse

cwd = Path().cwd()


def transform_coordinates(coords, scale_factor):
    x_center, y_center, width, height = coords
    x_center *= scale_factor
    y_center *= scale_factor
    width *= scale_factor
    height *= scale_factor
    return [x_center, y_center, width, height]


def apply_blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def apply_rescale(image, scale):
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)


def create_variants(index: int, image_file_path: Path, output_dir_path: Path):
    original_image = cv2.imread(image_file_path.as_posix())
    blur = [None, 2, 3]
    scale = [0.7, 1, 1.3]

    count = 1
    for i, blur_level in enumerate(blur):
        for j, scale_level in enumerate(scale):
            variation_image = original_image.copy()
            if blur_level:
                variation_image = apply_blur(variation_image, blur_level)
            variation_image = apply_rescale(variation_image, scale_level)

            rand = random.randint(0, 8)
            if rand == 0:
                folder = "validation"
            else:
                folder = "training"

            output_path_image = output_dir_path.joinpath(
                "images", folder, f"{index}_v{count}.png")
            cv2.imwrite(output_path_image.as_posix(), variation_image)

            output_path_file = output_dir_path.joinpath(
                "labels", folder, f"{index}_v{count}.txt")
            shutil.copy(image_file_path.as_posix().replace(
                'png', 'txt'), output_path_file.as_posix())
            count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Create a dataset to train a model from a collection of Images with their labels.")

    parser.add_argument("input_directory",
                        help="Input Directory")
    parser.add_argument("output_directory",
                        help="Output Directory")

    args = parser.parse_args()

    input_path = Path(args.input_directory)
    if not input_path.is_absolute():
        input_path = cwd.joinpath(input_path)

    output_path = Path(args.output_directory)
    if not output_path.is_absolute():
        output_path = cwd.joinpath(output_path)

    files = os.listdir(input_path.as_posix())
    image_names = [name for name in files if name.endswith(".png")]
    os.makedirs(f"{output_path.as_posix()}/images/training", exist_ok=True)
    os.makedirs(f"{output_path.as_posix()}/images/validation", exist_ok=True)
    os.makedirs(f"{output_path.as_posix()}/labels/training", exist_ok=True)
    os.makedirs(f"{output_path.as_posix()}/labels/validation", exist_ok=True)
    for index, image_name in enumerate(image_names):
        image_file_path = input_path.joinpath(image_name)
        create_variants(index, image_file_path, output_path)


if __name__ == "__main__":
    main()
