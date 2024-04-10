import os
import numpy as np
from PIL import Image

# Global variable to hold the number of iterations for certain functions
M = 0

def setM():
    global M
    M = int(input("Enter the number of iterations for subsequent functions: "))

def upscaleImage(image_np, original_filepath):
    """
    Upscales the given image to a new width, maintaining aspect ratio.

    :param image_np: Numpy array representation of the image to upscale.
    :param original_filepath: The file path of the original image for reference (not used in this version).
    """
    print(f"Original dimensions: {image_np.shape[1]}x{image_np.shape[0]}")
    new_width = int(input("Enter new width: "))
    aspect_ratio = image_np.shape[0] / image_np.shape[1]
    new_height = int(new_width * aspect_ratio)

    image = Image.fromarray(image_np)
    upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)  # High-quality downsampling

    return np.array(upscaled_image)


def process_image():
    filepath, original_image_np = load_and_prepare_image()
    if filepath is None or original_image_np is None:
        print("No image was loaded. Exiting the current process.")
        return

    # Ensure setM is called to define the global M
    setM()

    # Upscale the image
    upscaled_image_np = upscaleImage(original_image_np, filepath)

    # Save the upscaled image
    new_width = upscaled_image_np.shape[1]
    new_height = upscaled_image_np.shape[0]
    save_image(Image.fromarray(upscaled_image_np), filepath, new_width, new_height, "upscaled")
    imageSlice(upscaled_image_np, filepath)
    imageMask(upscaled_image_np, filepath)
    # Here, you'd pass copies of the upscaled image to other functions
    # For example: processed_image = some_function(upscaled_image_np.copy())
    # Note: Implement the actual image processing functions as needed.



def rotateImage(image):
    # Placeholder for the actual implementation
    return image


def imageSlice(image_np, original_filepath):
    """
    Edits a copy of the upscaled image, performing slicing operations (L, R, U, D) and saving results
    into their respective subdirectories named after the operations.

    :param image_np: Numpy array representation of the upscaled image.
    :param original_filepath: The file path for naming and saving the edited images.
    """
    original_size = image_np.shape[1], image_np.shape[0]  # Width, Height

    # Create a base folder for saving slice-modified images
    base_filename = os.path.basename(original_filepath)
    base_folder_name = f"{base_filename}_sliced"
    os.makedirs(base_folder_name, exist_ok=True)

    N = M  # Local copy of global M
    while N >= 1:
        for op in ["L", "R", "U", "D"]:
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
                    start_pos += 2 * N
            else:  # Operation on rows
                while start_pos < height:
                    image_copy = np.delete(image_copy, slice(start_pos, min(start_pos + N, height)), axis=0)
                    start_pos += 2 * N

            # After columns or rows are removed, resize to original dimensions
            result_image = Image.fromarray(image_copy).resize(original_size, Image.LANCZOS)

            # Save the file into the respective operation subdirectory
            filename = f"{base_filename}_{op}_{N}.png"
            result_image.save(os.path.join(op_folder, filename))

        N -= 1


def imageMask(image_np, original_filepath):
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
            op_folder = os.path.join(base_folder_name, op)
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

            # Save the file into the respective operation subdirectory
            filename = f"{base_filename}_{op}_{N}.png"
            result_image.save(os.path.join(op_folder, filename))

        N -= 1


# Function to load and prepare the image
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



def save_image(image, original_filepath, new_width, new_height, processing_step=""):
    """
    Saves the image in an uncompressed WebP format in a directory named after the original file base name.
    The directory is created in the same location as the original image. The saved file name includes the type
    of processing done, allowing for easy identification of the processing steps applied.

    :param image: The PIL Image object to save.
    :param original_filepath: The file path of the original image.
    :param new_width: The new width of the image.
    :param new_height: The new height of the image.
    :param processing_step: A string describing the processing step applied to the image.
    """
    file_dir, file_name = os.path.split(original_filepath)
    file_base, _ = os.path.splitext(file_name)
    new_directory = os.path.join(file_dir, file_base)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Adjusting filename based on processing step
    suffix = f"_{processing_step}" if processing_step else ""
    new_filename = f"{file_base}_{new_width}x{new_height}{suffix}.webp"
    image.save(os.path.join(new_directory, new_filename), format='WEBP', lossless=True)
    print(f"Image saved to {os.path.join(new_directory, new_filename)}")


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
