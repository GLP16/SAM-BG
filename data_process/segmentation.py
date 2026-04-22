from PIL import Image
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def split_image(image_path, output_folder, tile_size=256):
    """
    Split a large image into multiple smaller images of a specified size and name them in order as 0,1,2,...

    :param image_path: Path to the large image
    :param output_folder: Folder for output small images
    :param tile_size: Size of each small image (default is 256)
    """
    if not os.path.isfile(image_path):
        print(f"Error: Cannot find image file {image_path}")
        return

    try:
        image = Image.open(image_path)
        print(f"Successfully opened image file: {image_path}")
    except IOError:
        print(f"Error: Cannot open image file {image_path}")
        return

    image_width, image_height = image.size
    print(f"Large image size: {image_width}x{image_height}")

    cols = image_width // tile_size
    rows = image_height // tile_size

    if image_width % tile_size != 0:
        cols += 1
    if image_height % tile_size != 0:
        rows += 1

    print(f"Split image into {rows} rows and {cols} columns, total {rows * cols} tiles")

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except OSError as e:
            print(f"Error: Cannot create output folder {output_folder}. Error message: {e}")
            return
    else:
        print(f"Output folder already exists: {output_folder}")

    count = 0

    for row in range(rows):
        for col in range(cols):
            left = col * tile_size
            upper = row * tile_size
            right = min(left + tile_size, image_width)
            lower = min(upper + tile_size, image_height)

            bbox = (left, upper, right, lower)
            tile = image.crop(bbox)

            tile_filename = os.path.join(output_folder, f"{count}.png")
            try:
                tile.save(tile_filename)
                print(f"Saved tile: {tile_filename}")
                count += 1
            except IOError:
                print(f"Error: Cannot save tile {tile_filename}")

    print("Image splitting completed!")

if __name__ == "__main__":
    input_image_paths = [
        "",
        "",
    ]

    output_dirs = [
        "",
        ""
    ]

    if len(input_image_paths) != len(output_dirs):
        print("Error: The number of input image paths and output folders does not match. Please ensure they are the same.")
    else:
        for img_path, out_dir in zip(input_image_paths, output_dirs):
            print(f"/nProcessing image: {img_path}")
            split_image(img_path, out_dir, tile_size=256)
