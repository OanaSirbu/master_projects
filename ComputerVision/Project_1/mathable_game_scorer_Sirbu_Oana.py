# importing the necessary libraries
import cv2
import numpy as np
import os
import glob


# PART 0. DEFINING THE PATHS AND VARIABLES
# base directory is the main folder, in which one can find the images of the games and also 
# the folders with the results; this should be adjusted accordingly to the desired path
base_dir = "/home/oana/Documents/master/1st_year/CV/project_1/"

# the paths to the images of the board and tokens provided for this project, and the chosen template image
base_image_path = base_dir + "board+tokens/board1.jpg"
tokens_image_path = base_dir + "board+tokens/tokens1.jpg"
template_path = base_dir + "train/1_03.jpg"

# the following paths will have to be changed when we need to align test game images
# depending on where we want to store the processed images 
# games_images_folder represents the folder in which we will have all the images for the 4 games
games_images_folder = base_dir + "train"
# the next 3 folders will be created by the script at the right moment,
# we just need to specify the path where we want them to be created. they will contain processed images
aligned_images_folder = base_dir + "final/aligned_images"
cropped_folder = base_dir + "final/cropped_images"
final_boards_folder = base_dir + "final/final_boards"
# in the tokens folder we will find images with all the possible tokens, cropped and processed
tokens_folder = base_dir + "tokens"

# this is the folder where the results will be stored (also may need adjustment)
submission_folder = base_dir + "final/407_Sirbu_Oana-Adriana"
os.makedirs(submission_folder, exist_ok=True)


# predefined variable NUM_ROUNDS as I know we will always have 50 rounds for each game
NUM_ROUNDS = 50

# this may need to be changed, depending on the game numbers of the evaluation part
game_numbers = [1, 2, 3, 4]
# for the fake test I used this:
# game_numbers = [1]


# PART I. ALIGN ALL IMAGES 

# simple function to get all the images paths for a game
def get_images_for_game(game_number, directory):
    game_images = []
    pattern = os.path.join(directory, f"{game_number}_*.jpg")
    for file_path in sorted(glob.glob(pattern)):
        game_images.append(file_path)
    return game_images


# the following functions are used to preprocess the images
# the first one is used to enhance the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


# the following functions are used to binarize the image. there are 2 of them because
# I couldn't get the best results with only one of them (still had to deal with noise)
def binarize_image(image):
    otsu_threshold_source, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresholded_image = cv2.threshold(image, otsu_threshold_source, 255, cv2.THRESH_BINARY)
    return thresholded_image


def remove_noise(image):
    _, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    return binary_image


def blur_image(image, blur_amount=5):
    return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)


# function to load images from a folder
def load_images_from_folder(folder):
    images = {}
    filenames = os.listdir(folder)
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[filename] = img
    return images


# this class is inspired from the laboratory and it is
# used to align the images to the template image using SIFT
class GameBoard:
    def __init__(self, template_image):
        self._template_image = template_image

    def _get_keypoints_and_features(self, image) -> tuple:
        sift = cv2.SIFT_create() 
        keypoints = sift.detect(image, None) 
        keypoints, features = sift.compute(image, keypoints) 
        return keypoints, features

    def _generate_homography(self, all_matches, keypoints_source, keypoints_dest, ratio = 0.75, ransac_rep = 4.0):
        if not all_matches:
            return None
        matches = [] 
        for match in all_matches:  
            if len(match) == 2 and (match[0].distance / match[1].distance) < ratio:
                matches.append(match[0])
        points_source = np.float32([keypoints_source[m.queryIdx].pt for m in matches]) 
        points_dest = np.float32([keypoints_dest[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(points_source, points_dest, cv2.RANSAC, ransac_rep)
        return H

    def _match_features(self, features_source, features_dest):
        feature_matcher = cv2.DescriptorMatcher_create("FlannBased")
        matches = feature_matcher.knnMatch(features_source, features_dest, k=2)   
        return matches

    def align_and_save(self, source_image, output_path):
        aligned_image = self.scale_image_to_template(source_image)
        cv2.imwrite(output_path, aligned_image)

    def scale_image_to_template(self, image):
        keypoints_source, features_source = self._get_keypoints_and_features(image)
        keypoints_dest, features_dest = self._get_keypoints_and_features(self._template_image)
        all_matches = self._match_features(features_source, features_dest)
        H = self._generate_homography(all_matches, keypoints_source, keypoints_dest, 0.75, 4)
        result = cv2.warpPerspective(image, H, (self._template_image.shape[1], self._template_image.shape[0]))
        return result


# dictionary to store the images for each game; game number will be the key
# and the lists with paths to the images for that game will be the values
game_images_dict = {}

for game_number in game_numbers:
    game_images = get_images_for_game(game_number, games_images_folder)
    game_images_dict[game_number] = game_images
    print(f"Images for {game_number} have been processed. You can further use them")


# preprocess the template path and create the GameBoard object based on this template
initial_template = preprocess_image(template_path)
game_board_template = GameBoard(initial_template)


# we create the folder in which we store all the aligned images 
os.makedirs(aligned_images_folder, exist_ok=True)

print("Started alignment. It may take a while.")
# base image is considered an image with the board having no tiles on it; we align such an image too
base_image = preprocess_image(base_image_path)
game_board_template.align_and_save(base_image, os.path.join(aligned_images_folder, 'base_aligned.jpg'))


# align all the images for each game
for game_number, game_images in game_images_dict.items():
    for image_path in game_images:
        source = preprocess_image(image_path)

        filename = os.path.basename(image_path)
        print(filename)

        aligned_output_path = os.path.join(aligned_images_folder, filename)
        game_board_template.align_and_save(source, aligned_output_path)

print("Alignment completed. We will proceed with cropping the images.")

# at this point, we should have all the images aligned and saved in the aligned_images_folder


# PART II. CROPPING THE IMAGES (such that we keep only the grid from the board)
# --- this part will be explained better in the report, as it is a bit more complex and will not be directly used in the evaluation part ---

# clicked_points = []
# display_image = initial_template.copy()


# corner_coordinates = []

# def select_corners(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(corner_coordinates) < 4:  
#             corner_coordinates.append((x, y))
#             cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow('Select Corners', display_image)

#             if len(corner_coordinates) == 4:
#                 cv2.line(display_image, corner_coordinates[0], corner_coordinates[1], (0, 255, 0), 2)
#                 cv2.line(display_image, corner_coordinates[1], corner_coordinates[2], (0, 255, 0), 2)
#                 cv2.line(display_image, corner_coordinates[2], corner_coordinates[3], (0, 255, 0), 2)
#                 cv2.line(display_image, corner_coordinates[3], corner_coordinates[0], (0, 255, 0), 2)
#                 cv2.imshow('Select Corners', display_image)


# cv2.namedWindow('Select Corners', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('Select Corners', select_corners)

# cv2.imshow('Select Corners', display_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("Selected Corner Coordinates:")
# for i, corner in enumerate(corner_coordinates):
    # print(f"Corner {i+1}: {corner}")


# the coordinates of the corners of the board are hardcoded, to faciliate the process;
# they were obtained through the method above
corner_coordinates = [(1075, 516), (3017, 510), (3052, 2455), (1075, 2468)]

# define the width and height of the output image with the grid cells
output_width = 1400
output_height = 1400

# define the corners of the output image
output_corners = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
# this function computes a perspective transformation matrix based on these corner coordinates and the desired output corners
transformation_matrix = cv2.getPerspectiveTransform(np.array(corner_coordinates, dtype=np.float32), output_corners)
# now we apply the perspective transformation defined by the transformation matrix to the initial template
cropped_template = cv2.warpPerspective(initial_template, transformation_matrix, (output_width, output_height))


# if the cropped folder doesn't exist, we create it
os.makedirs(cropped_folder, exist_ok=True)

print("Cropping the images. It should be a fast process.")

base_image = cv2.imread(os.path.join(aligned_images_folder, 'base_aligned.jpg'), cv2.IMREAD_GRAYSCALE)
# after loading the base image, we remove the noise, crop and binarize it using the custom functions presented previously
improved_base_image = remove_noise(base_image)
cropped_base_image = cv2.warpPerspective(improved_base_image, transformation_matrix, (output_width, output_height))
binarized_base_image = binarize_image(cropped_base_image)
cv2.imwrite(os.path.join(cropped_folder, 'base_aligned.jpg'), cropped_base_image)

# we use the same approach for all the images in the aligned_images_folder
for filename in os.listdir(aligned_images_folder):
    if filename.endswith('.jpg'): 
        input_image = cv2.imread(os.path.join(aligned_images_folder, filename), cv2.IMREAD_GRAYSCALE)
        improved_input_image = remove_noise(input_image)
        cropped_image = cv2.warpPerspective(improved_input_image, transformation_matrix, (output_width, output_height))
        # result_image = binarize_image(cropped_image)

        output_path = os.path.join(cropped_folder, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"Processed: {filename}")

print("All images cropped and saved.")


# here I identified the rows and columns of the board; it will be commented such that it will save a lot of time
# to simply store the results in a variable and use them in the next steps
# I couldn't automate this process because the widths and the heights of the cells are not always the same in my case

# row_starts = []
# max_rows = 15  

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN and len(row_starts) < max_rows:
#         row_starts.append(y)
#         if len(row_starts) == max_rows:
#             print("Maximum number of rows reached. Stop selecting row start points.")

# cv2.namedWindow("Select Rows", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("Select Rows", mouse_callback)

# while True:
#     cv2.imshow("Select Rows", base_image)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or len(row_starts) == max_rows:  
#         break
#     elif key == ord('c'):  
#         row_starts = []


# cv2.destroyAllWindows()
# print("Selected Row Start Points:", row_starts)


# column_starts = []
# max_cols = 15  

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN and len(column_starts) < max_cols:
#         column_starts.append(x)
#         if len(column_starts) == max_cols:
#             print("Maximum number of cols reached. Stop selecting cols start points.")


# cv2.namedWindow("Select columns", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("Select columns", mouse_callback)

# while True:
#     cv2.imshow("Select columns", base_image)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or len(column_starts) == max_cols:  
#         break
#     elif key == ord('c'):  
#         row_starts = []

# cv2.destroyAllWindows()
# print("Selected Cols Start Points:", column_starts)


# the rows and columns of the grid board are hardcoded, to faciliate the process; they were obtained through the method above
row_starts = [0, 98, 197, 298, 397, 498, 599, 697, 797, 898, 999, 1097, 1199, 1299, 1399]
column_starts = [0, 111, 209, 310, 408, 507, 608, 706, 801, 902, 1002, 1100, 1201, 1300, 1399]

# list to store the boundaries of the cells; each cell will be identified through the top left corner and the bottom right corner
grid_cells = []

for i in range(len(row_starts) - 1):
    for j in range(len(column_starts) - 1):
        cell_boundary = ((column_starts[j], row_starts[i]), (column_starts[j + 1], row_starts[i + 1]))
        grid_cells.append(cell_boundary)

# we can print the boundaries of the cells to see if they are correct
for idx, cell_boundary in enumerate(grid_cells, 1):
    print(f"Cell {idx}: {cell_boundary}")


# if we want to see the grid overlapping with the board:

# aligned_img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
# grid_color = (0, 255, 0)  

# for cell_boundary in grid_cells:
#     top_left = cell_boundary[0]
#     bottom_right = cell_boundary[1]
    
#     cv2.line(aligned_img, top_left, (bottom_right[0], top_left[1]), grid_color, 2)
#     cv2.line(aligned_img, (top_left[0], bottom_right[1]), bottom_right, grid_color, 2)
    
#     cv2.line(aligned_img, top_left, (top_left[0], bottom_right[1]), grid_color, 2)
#     cv2.line(aligned_img, (bottom_right[0], top_left[1]), bottom_right, grid_color, 2)

# cv2.namedWindow("Aligned Image with Grid", cv2.WINDOW_NORMAL)
# cv2.imshow("Aligned Image with Grid", aligned_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# function to get the row and column of a cell based on the coordinates of the top left and bottom right corners
# rows are numbered from 1 to 14 and columns from A to N
def get_row_and_column(x, y, grid_cells):
    cell_width = (grid_cells[1][1][0] - grid_cells[1][0][0])
    cell_height = (grid_cells[1][1][1] - grid_cells[1][0][1])

    # check if x coordinate falls within the first column (as it is the problematic one because of the different width)
    if x < grid_cells[1][0][0]:
        column_index = 0
    else:
        column_index = int((x - grid_cells[1][0][0]) // cell_width) + 1
    
    row_index = int((y - grid_cells[0][0][1]) // cell_height) + 1
    
    column_letter = chr(ord('A') + column_index)
    row_number = row_index
    
    return f"{row_number}{column_letter}"


# function to get the coordinates of the top left and bottom right corners of a cell based on the row and column
# this is the reverse of the previous function; both of them proved to be useful in the next steps
def get_grid_cell_coordinates(row, column, grid_cells):
    row_index = row - 1 
    column_index = column - 1 
    
    cell_index = row_index + column_index * 14 
    cell_coordinates = grid_cells[cell_index]
    
    top_left = cell_coordinates[0]
    bottom_right = cell_coordinates[1]
    
    return top_left, bottom_right


# PART III. EXTRACT TOKENS
# I will extract images of all possible tokens in order to make template matching later

# to do this, we need first to align and crop an image containing all possible tokens
print("Aligning and cropping the tokens image.")
tokens_image = preprocess_image(tokens_image_path)

aligned_tokens_path = os.path.join(aligned_images_folder, 'tokens_aligned.jpg')
game_board_template.align_and_save(tokens_image, aligned_tokens_path)

tokens_aligned_image = cv2.imread(os.path.join(aligned_images_folder, 'tokens_aligned.jpg'), cv2.IMREAD_GRAYSCALE)
improved_tokens_image = remove_noise(tokens_aligned_image)
cropped_tokens_image = cv2.warpPerspective(improved_tokens_image, transformation_matrix, (output_width, output_height))
cropped_tokens_image = blur_image(cropped_tokens_image)

cv2.imwrite(os.path.join(cropped_folder, 'tokens_aligned.jpg'), cropped_tokens_image)


# after aligning and cropping the tokens image, we can extract the tokens based on the cells
# in which the tiles are placed; we will save the tokens in the tokens folder
def extract_tokens(board_image, grid_cells, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # create a list with the possible tokens
    tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 
              27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]
    
    i = 0
    
    row_end = 12 
    for row in range(5, 11): 
        for column in range(4, row_end):  
            if column in [10, 11] and row == 10:
                pass
            else:
                top_left, bottom_right = get_grid_cell_coordinates(column, row, grid_cells)
                
                grid_cell = board_image[top_left[1]+4:bottom_right[1], top_left[0]+9:bottom_right[0]-4]
                
                gray_cell = cv2.cvtColor(grid_cell, cv2.COLOR_BGR2GRAY)
                
                # threshold the grid cell to obtain a binary image
                _, thresh = cv2.threshold(gray_cell, 200, 255, cv2.THRESH_BINARY)
                
                # find the coordinates of the bounding box around the white content
                coords = cv2.findNonZero(thresh)
                x, y, w, h = cv2.boundingRect(coords)
                
                # crop the grid cell using the coordinates of the bounding box
                cropped_cell = grid_cell[y:y+h, x:x+w]
                
                # save the cropped cell as a token image
                token_filename = f"token_{tokens[i]}.jpg"  
                token_path = os.path.join(output_folder, token_filename)
                cv2.imwrite(token_path, cropped_cell)
                i += 1
    return tokens


tokens_image = cv2.imread(os.path.join(cropped_folder, 'tokens_aligned.jpg'))
extract_tokens(tokens_image, grid_cells, tokens_folder)

print("Tokens extracted and saved.")


# PART IV. PROCESSING THE FINAL IMAGES
def invert_center_cells(image, grid_cells):
    # Invert the binary values of the four centered cells
    # if we don't perform this step, we will have problems with the template matching (it will identify digits in the center region)
    center_cells = [(7, 7), (7, 8), (8, 7), (8, 8)]
    for row, col in center_cells:
        top_left, bottom_right = get_grid_cell_coordinates(row, col, grid_cells)
        center_cell = image[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1]
        center_cell = cv2.bitwise_not(center_cell)
        image[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1] = center_cell
    
    return image


print("Processing the final images.")


base = np.copy(base_image)
base = invert_center_cells(base.copy(), grid_cells)
# cv2.namedWindow('Image with Center Cells inverted', cv2.WINDOW_NORMAL)
# cv2.imshow('Image with Center Cells inverted', base)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(cropped_folder + '/base_inverted_cells.jpg', base)


# we need to apply the previous approach to all the images in the cropped folder
# such that we won't have problems with wrong matches
# this step implies inverting the center cells for all cropped images without overwriting the original images
# so they will be store in the final_boards_folder

for filename in os.listdir(cropped_folder):
    if filename.endswith('.jpg') and filename not in ['tokens_aligned.jpg', 'base_inverted_cells.jpg']:
        image_path = os.path.join(cropped_folder, filename)
        
        image = cv2.imread(image_path)
        
        processed_image = invert_center_cells(image, grid_cells)
        processed_image = blur_image(processed_image)
        
        output_folder = final_boards_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)

print("All final images processed and saved.")


# we load the images from the tokens folder to use them in the template matching part
tile_images = load_images_from_folder(tokens_folder)


# function to update the board image for each round
def update_board_image(round_number, game_number):
    if round_number < 10:
        board_image_path = f"{final_boards_folder}/{game_number}_0{round_number}.jpg"
    else:
        board_image_path = f"{final_boards_folder}/{game_number}_{round_number}.jpg"
    if os.path.isfile(board_image_path):
        return cv2.imread(board_image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return None
    

# function to mark the visited cells on the board image; otherwise we risk to select the same cell multiple times
def mark_visited_cells(image, visited_cells, grid_cells):
    # Copy the image to avoid modifying the original
    marked_image = image.copy()

    for cell in visited_cells:
        row, col = cell_to_coordinates(cell)
        top_left, bottom_right = get_grid_cell_coordinates(col, row, grid_cells)
        cv2.rectangle(marked_image, top_left, bottom_right, (0, 0, 0), -1)  # -1 fills the rectangle

    return marked_image


# simple function to convert the cell name (string) to row and column (integers)
# example: '1A' -> (1, 1)
def cell_to_coordinates(cell):
    row = int(cell[:-1])
    col = ord(cell[-1]) - ord('A') + 1
    return row, col


print("Starting template matching.")
# PART V. TEMPLATE MATCHING

# the main function to perform the template matching; its behaviour will pe explained in the PDF documentation
def main():
    threshold = 0.972
    grid_rows = 14
    grid_cols = 'N'

    # define function to get neighboring cells
    def get_neighbors(cell):
        row = int(cell[:-1])
        col = cell[-1]
        neighbors = set()
        for i in range(row - 1, row + 2):
            for j in range(ord(col) - 1, ord(col) + 2):
                if 1 <= i <= grid_rows and ord('A') <= j <= ord(grid_cols):
                    neighbors.add(f"{i}{chr(j)}")
        return neighbors

    for game_number in range(1, 5):
        visited_cells = {'7G', '7H', '8G', '8H'}
        high_scoring_tiles = []  # list to store high-scoring tiles

        for round_number in range(1, NUM_ROUNDS + 1):
            board_image = update_board_image(round_number, game_number=game_number)
            if board_image is None:
                print(f"Board image not found for round {round_number}.")
                continue
            
            modified_board_image = mark_visited_cells(board_image, visited_cells=visited_cells, grid_cells=grid_cells)
            
            sorted_tile_images = dict(sorted(tile_images.items(), key=lambda item: int(item[0].split('.')[0].split('_')[1]), reverse=True))
            sorted_items = list(sorted_tile_images.items())

            sorted_items[-1], sorted_items[-2] = sorted_items[-2], sorted_items[-1]

            sorted_tile_images = dict(sorted_items)

            neighbors = set()

            for cell in visited_cells:
                neighbors.update(get_neighbors(cell))

            neighboring_cells = neighbors - visited_cells

            # reset high_scoring_tiles list for each round
            high_scoring_tiles.clear()

            # we want to process double-digit numbers first
            for tile_filename, tile_img in sorted_tile_images.items():
                tile_number = tile_filename.split('.')[0].split('_')[1]
                # check if the token has two digits
                if len(tile_number) == 2:
                    for angle in [0, -5, 5]:
                        rotated_tile_img = cv2.warpAffine(tile_img, cv2.getRotationMatrix2D((tile_img.shape[1] // 2, tile_img.shape[0] // 2), angle, 1), (tile_img.shape[1], tile_img.shape[0]))

                        result = cv2.matchTemplate(modified_board_image, rotated_tile_img, cv2.TM_CCORR_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        # print(tile_filename, max_val)

                        if max_val > threshold:
                            high_scoring_tiles.append((max_val, max_loc, tile_number))

            # if no high-scoring tiles found among double-digit numbers, we check single-digit numbers
            if not high_scoring_tiles:
                for tile_filename, tile_img in sorted_tile_images.items():
                    tile_number = tile_filename.split('.')[0].split('_')[1]
                    # check if the token has one digit
                    if len(tile_number) == 1:
                        for angle in [0, -5, 5]:
                            rotated_tile_img = cv2.warpAffine(tile_img, cv2.getRotationMatrix2D((tile_img.shape[1] // 2, tile_img.shape[0] // 2), angle, 1), (tile_img.shape[1], tile_img.shape[0]))

                            result = cv2.matchTemplate(modified_board_image, rotated_tile_img, cv2.TM_CCORR_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)
                            # print(tile_filename, max_val)

                            if max_val > threshold:
                                cell_top_left, cell_bottom_right = max_loc[0] + 15, max_loc[1]
                                cell = get_row_and_column(cell_top_left, cell_bottom_right, grid_cells)

                                if cell in neighboring_cells and cell not in visited_cells:
                                    round_number_str = str(round_number).zfill(2)
                                    filename = f"{submission_folder}/{game_number}_{round_number_str}.txt"
                                    with open(filename, 'w') as f:
                                        f.write(f"{cell} {tile_number}\n")
                                    visited_cells.add(cell)
                                    break
                        else:
                            continue
                        break
                else:
                    print("No valid match found for round:", round_number)
            else:
                # we choose the tile with the highest score among high-scoring tiles
                # print(high_scoring_tiles)
                max_score_tile = max(high_scoring_tiles, key=lambda x: x[0])
                max_val, max_loc, tile_number = max_score_tile
                cell_top_left, cell_bottom_right = max_loc[0] + 10, max_loc[1]
                cell = get_row_and_column(cell_top_left, cell_bottom_right, grid_cells)

                if cell in neighboring_cells and cell not in visited_cells:
                    round_number_str = str(round_number).zfill(2)
                    filename = f"{submission_folder}/{game_number}_{round_number_str}.txt"
                    with open(filename, 'w') as f:
                        f.write(f"{cell} {tile_number}\n")
                    visited_cells.add(cell)


main()
print("Template matching completed.")


# PART VI. HELPER FUNCTIONS FOR SCORING

# function to read information about game rounds from text files in a given folder
# and store the data in a dictionary where keys are game numbers and values are lists of round data tuples
def read_round_info(results_folder):
    all_positions_and_tokens = {}
    for filename in sorted(os.listdir(results_folder)):
        if filename.endswith(".txt") and not filename.endswith("scores.txt"):
            game_number = filename.split('_')[0].split('.')[0]
            if len(game_number) == 2 and game_number.startswith('0'):
                game_number = game_number[1:]
            game_number = int(game_number)
            file_path = os.path.join(results_folder, filename)
            positions_and_tokens = []
            with open(file_path, 'r') as file:
                for line in file:
                    round_data = line.strip().split()
                    print(round_data)
                    round_data = tuple([round_data[0], int(round_data[1])])
                    positions_and_tokens.append(round_data)
            all_positions_and_tokens.setdefault(game_number, []).extend(positions_and_tokens)
    return all_positions_and_tokens


positions_and_tokens = read_round_info(submission_folder)
print(positions_and_tokens)


# function to read player turns data from text files in a specified folder
# and organize the data into a dictionary where keys are game numbers
# and values are lists of tuples containing player turn information
def read_player_turns(folder):
    all_player_turns = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith("turns.txt"):
            game_number = filename.split('_')[0].split('.')[0]
            if len(game_number) == 2 and game_number.startswith('0'):
                game_number = game_number[1:]
            game_number = int(game_number)
            file_path = os.path.join(folder, filename)
            players_turns = []
            with open(file_path, 'r') as file:
                for line in file:
                    turns_data = line.strip().split()
                    turns_data = tuple([turns_data[0], int(turns_data[1])])
                    players_turns.append(turns_data)
            all_player_turns.setdefault(game_number, []).extend(players_turns)
    return all_player_turns


player_turns = read_player_turns(games_images_folder)
print(player_turns)


def create_mathable_board():
    # we initialize an empty board with 14 rows and 14 columns
    board = [['x' for _ in range(14)] for _ in range(14)]

    # define the positions of special places and operands
    # as tuples of rows and columns
    double_score_places = [(2, 2), (2, 13), (3, 3), (3, 12), (4, 4), (4, 11), (5, 5), (5, 10),
                           (10, 5), (10, 10), (11, 4), (11, 11), (12, 3), (12, 12), (13, 2), (13, 13)]
    triple_score_places = [(1, 1), (1, 14), (14, 1), (14, 14), (1, 7), (1, 8), (7, 1), (7, 14), (8, 1), (8, 14), (14, 7), (14, 8)]
    substract_sign = [(3, 6), (3, 9), (6, 3), (6, 12), (9, 3), (9, 12), (12, 6), (12, 9)]
    addition_sign = [(4, 7), (5, 8), (7, 5), (8, 4), (7, 11), (8, 10), (10, 7), (11, 8)]
    multiplication_sign = [(4, 8), (5, 7), (7, 4), (8, 5), (7, 10), (8, 11), (10, 8), (11, 7)]
    division_sign = [(2, 5), (2, 10), (5, 2), (5, 13), (10, 2), (10, 13), (13, 5), (13, 10)]

    # then place special places and operands on the board
    for row, col in double_score_places:
        board[row-1][col-1] = '*2'
    for row, col in triple_score_places:
        board[row-1][col-1] = '*3'
    for row, col in substract_sign:
        board[row-1][col-1] = '-'
    for row, col in addition_sign: 
        board[row-1][col-1] = '+'
    for row, col in multiplication_sign:
        board[row-1][col-1] = '*'
    for row, col in division_sign:
        board[row-1][col-1] = '/'

    # include the center tokens as well
    board[6][6] = 1
    board[6][7] = 2
    board[7][6] = 3
    board[7][7] = 4

    return board


# we create such a board, without any tile on it
board = create_mathable_board()
for row in board:
    print(' '.join(str(cell) for cell in row))


# function to evaluate an expression on the board; check the mathematical operations from which the token is obtained 
def evaluate_expression(board, row1, col1, operand, row2, col2):
    try:
        if row1 < 0 or row1 >= len(board) or col1 < 0 or col1 >= len(board[0]) \
                or row2 < 0 or row2 >= len(board) or col2 < 0 or col2 >= len(board[0]):
            return None
        value1 = float(board[row1][col1])
        value2 = float(board[row2][col2])
    except (ValueError, IndexError):
        return None

    if operand == '+':
        result1 = value1 + value2
        result2 = value2 + value1
    elif operand == '-':
        result1 = value1 - value2
        result2 = value2 - value1
    elif operand == '*':
        result1 = value1 * value2
        result2 = value2 * value1
    elif operand == '/':
        if value2 == 0 and value1 == 0:
            return None
        elif value2 == 0:
            result1 = None
            result2 = 0
        elif value1 == 0:
            result1 = 0  
            result2 = None
        else:
            result1 = value1 / value2
            result2 = value2 / value1

    return result1, result2


# compute the score of a single round; here we check multiple situations:
# if we have a tile placed on a mathematical symbol, if we have a tile placed on a score multiplier cell etc
def compute_round_score(old_board, new_board, cell, token):
    score = 0
    row, col = cell_to_coordinates(cell)
    cell_value = old_board[row-1][col-1]
    new_board[row-1][col-1] = token

    if cell_value in ['+', '-', '*', '/']:
        return token

    pair_operations = [(row-2, col-1, row-3, col-1),
                      (row, col-1, row+1, col-1),
                      (row-1, col-2, row-1, col-3),
                      (row-1, col, row-1, col+1)]

    for pair_row1, pair_col1, pair_row2, pair_col2 in pair_operations:
        for operand in ['+', '-', '*', '/']:
            results = evaluate_expression(new_board, pair_row1, pair_col1, operand, pair_row2, pair_col2)
            if results is not None:
                result1, result2 = results
                if result1 == token or result2 == token:
                    score += token
                    break  # break out of the loop for operands if any operation works for the pair

    if cell_value == '*2':
        score *= 2
    elif cell_value == '*3':
        score *= 3

    return score


# PART VII. COMPUTE PLAYER SCORE

# function to compute the score of a player for a single turn, but for all the rounds in a game
# we will also interate through all games available 
# and store the results in text files in the submissions folder
def compute_player_score(initial_board, positions_and_tokens, player_turns, output_folder):
    for game_number in range(1, 5):
        board = [row[:] for row in initial_board]
        turn = 0
        player_score = 0
        with open(f'{output_folder}/{game_number}_scores.txt', 'w') as file:
            for round_num in range(NUM_ROUNDS):
                cell = positions_and_tokens[game_number][round_num][0]
                token = positions_and_tokens[game_number][round_num][1]
                # print(round_num, cell, token)

                current_player = player_turns[game_number][turn][0]


                if turn == len(player_turns[game_number]) - 1:
                    change_players_turn = NUM_ROUNDS - 1
                else:
                    change_players_turn = player_turns[game_number][turn+1][1] - 1

                # print(change_players_turn)

                if round_num == NUM_ROUNDS-2: 
                    player_score += compute_round_score(initial_board, board, cell, token)
                elif round_num == NUM_ROUNDS-1:
                    player_score += compute_round_score(initial_board, board, cell, token)
                    file.write(f'{current_player} {player_turns[game_number][turn][1]} {player_score}')
                    break
                else:
                    if round_num == change_players_turn - 1:
                        player_score += compute_round_score(initial_board, board, cell, token)
                        file.write(f'{current_player} {player_turns[game_number][turn][1]} {player_score}\n')
                        turn += 1
                        player_score = 0
                    elif round_num < change_players_turn - 1:
                        player_score += compute_round_score(initial_board, board, cell, token)

    return "All rounds computed successfully"

print("Computing player scores.")
print(compute_player_score(board, positions_and_tokens, player_turns, submission_folder))