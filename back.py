import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Global variable to hold the number of iterations for certain functions
M = 0


def set_m():
    global M
    M = int(input("Enter the number of iterations for subsequent functions: "))


def upscale_image(image_np, original_filepath):
    """
    Upscales the given image to a new width, maintaining aspect ratio, and
    saves the image in an uncompressed WebP format in a directory named after
    the original file base name. The directory is created in the same location
    as the original image. The saved file name includes the type of processing
    done, allowing for easy identification of the processing steps applied.

    :param image_np: Numpy array representation of the image to upscale.
    :param original_filepath: The file path of the original image for reference.

    :return: A tuple containing the upscaled Numpy image array and the new folder location where the image is saved.
    """
    print(f"Original dimensions: {image_np.shape[1]}x{image_np.shape[0]}")

    # User input for new width; for automation or script use, replace the input with a direct argument
    new_width = int(input("Enter new width: "))
    aspect_ratio = image_np.shape[0] / image_np.shape[1]
    new_height = int(new_width * aspect_ratio)

    # Create the PIL image from the numpy array, upscale it, and convert back to numpy array
    image = Image.fromarray(image_np)
    upscaled_image_pil = image.resize((new_width, new_height), Image.LANCZOS)  # High-quality downsampling
    upscaled_image_np = np.array(upscaled_image_pil)
    # Determine the new file path
    file_dir, file_name = os.path.split(original_filepath)
    file_base, _ = os.path.splitext(file_name)
    new_directory = os.path.join(file_dir, f"{file_base}_{new_width}x{new_height}")
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Adjusting filename and saving the file
    new_filename = f"{file_base}_{new_width}x{new_height}.webp"
    upscaled_image_pil.save(os.path.join(new_directory, new_filename), format='WEBP', lossless=True)
    print(f"Image saved to {os.path.join(new_directory, new_filename)}")

    return upscaled_image_np, new_directory  # returning the numpy array and new folder location


def process_image():
    filepath, original_image_np = load_and_prepare_image()
    if filepath is None or original_image_np is None:
        print("No image was loaded. Exiting the current process.")
        return

    # Ensure setM is called to define the global M
    set_m()

    # Upscale the image
    upscaled_image_np, new_filepath = upscale_image(original_image_np, filepath)
    image_slice(upscaled_image_np, new_filepath)
    image_mask(upscaled_image_np, new_filepath)


def perform_slice(image_np, original_filepath, base_folder_name, op, N, original_size):
    """
    Function to perform the slicing operation on an image chunk and save the result.
    """
    image_copy = image_np.copy()
    width, height = image_copy.shape[1], image_copy.shape[0]

    # Create subdirectories for each operation within the base folder
    op_folder = os.path.join(base_folder_name, op)
    os.makedirs(op_folder, exist_ok=True)

    if op in ["L", "U"]:
        start_pos = 0
    else:
        start_pos = N

    if op in ["L", "R"]:  # Operation on columns
        while start_pos < width:
            image_copy = np.delete(image_copy, slice(start_pos, min(start_pos + N, width)), axis=1)
            start_pos += N
    else:  # Operation on rows
        while start_pos < height:
            image_copy = np.delete(image_copy, slice(start_pos, min(start_pos + N, height)), axis=0)
            start_pos += N

    # After columns or rows are removed, resize to original dimensions
    result_image = Image.fromarray(image_copy).resize(original_size, Image.LANCZOS)

    # Save the file into the respective operation subdirectory as uncompressed webp
    base_filename = os.path.basename(original_filepath)
    filename = f"{base_filename}_sliced_{op}_{N}.webp"
    result_image.save(os.path.join(op_folder, filename), format='webp', lossless=True)


def image_slice(image_np, original_filepath):
    """
    Enhanced version of the 'image_slice' using multiprocessing to speed up the
    slicing operations.
    """
    original_size = image_np.shape[1], image_np.shape[0]  # Width, Height

    # Create a base folder for saving slice-modified images
    base_filename = os.path.splitext(os.path.basename(original_filepath))[0]
    base_folder_name = f"{base_filename}_sliced"
    os.makedirs(base_folder_name, exist_ok=True)

    N = M  # Local copy of global M
    with ProcessPoolExecutor() as executor:
        while N >= 1:
            jobs = []
            for op in ["L", "R", "U", "D"]:
                job = executor.submit(perform_slice, image_np, original_filepath, base_folder_name, op, N, original_size)
                jobs.append(job)
            # Wait for all operations of the current iteration to complete
            for job in jobs:
                job.result()
            N -= 1


def image_mask(image_np, original_filepath):
    """
    Creates a mask over the given image, by setting pixels in specific rows/columns to (0,0,0,0) based on operations (L, R, U, D),
    and saves the results into their respective subdirectories named after the operations.

    :param image_np: Numpy array representation of the upscaled image (assumed to be in RGBA for transparency).
    :param original_filepath: The file path for naming and saving the edited images.
    """
    # Ensure the image is in RGBA mode for transparency handling
    if image_np.shape[2] == 3:  # RGB mode, convert to RGBA
        image_np = np.concatenate([image_np, 255 * np.ones((*image_np.shape[:2], 1), dtype=image_np.dtype)], axis=-1)

    original_size = image_np.shape[1], image_np.shape[0]  # Width, Height

    # Create a base folder for saving masked images
    base_filename = os.path.basename(original_filepath)

    base_folder_name = f"{base_filename}_masked"
    os.makedirs(base_folder_name, exist_ok=True)

    N = M  # Local copy of global M
    while N >= 1:
        for op in ["L", "R", "U", "D"]:
            image_copy = image_np.copy()

            # Create subdirectories for each operation within the base folder
            op_folder = os.path.join(base_folder_name, base_filename+'_'+'masked_'+op)
            os.makedirs(op_folder, exist_ok=True)

            if op in ["L", "U"]:
                start_pos = 0
            else:
                start_pos = N

            if op in ["L", "R"]:  # Masking columns
                for x_start in range(start_pos, image_copy.shape[1], 2 * N):
                    image_copy[:, x_start:x_start + N, :] = 0
            else:  # Masking rows
                for y_start in range(start_pos, image_copy.shape[0], 2 * N):
                    image_copy[y_start:y_start + N, :, :] = 0

            # After pixels are masked, resize to maintain original dimensions (if necessary)
            result_image = Image.fromarray(image_copy).resize(original_size, Image.LANCZOS)

            # Save the file into the respective operation subdirectory as uncompressed webp
            base_filename = os.path.basename(original_filepath)
            filename = f"{base_filename}_masked_{op}_{N}.webp"
            result_image.save(os.path.join(op_folder, filename), format='webp', lossless=True)

        N -= 1
def load_and_prepare_image():
    default_filepath = 'image.jpg'
    # Check if the default file exists in the current directory
    if os.path.exists(default_filepath):
        filepath = default_filepath
    else:
        # Prompt the user if the default file is not found
        filepath = input("Enter the file location: ")
        if not filepath or not os.path.exists(filepath):
            print("File not found. Please ensure the file path is correct.")
            return None, None

    try:
        # Load the image and convert it to RGBA to ensure it has a transparency layer.
        image = Image.open(filepath).convert("RGBA")
        original_image_np = np.array(image)
        return filepath, original_image_np
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None


# Main function to orchestrate the flow
def main():
    # Initial process call; can process the default 'image.jpg' or any user-specified image
    process_image()

    while True:
        another = input("Process another file? (yes/no): ").lower().strip()
        # Accept 'yes', 'y', or an empty string (just pressing Enter) as affirmative answers
        if another in ['yes', 'y', '']:
            process_image()  # Process the new image
        else:
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()
