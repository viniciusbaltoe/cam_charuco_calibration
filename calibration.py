import numpy as np
import cv2
import cv2.aruco as aruco
import os

# Define the folder containing the calibration images
images_folder = './etc/charuco_imgs/'

# ArUco dictionary type and board parameters
ARUCO_DICT_TYPE = aruco.DICT_4X4_50
BOARD_SIZE = (9, 7)
SQUARE_SIZE = 0.045
MARKER_SIZE = 0.035

def get_image_files(directory):
    # Function to retrieve a list of image files from the specified directory
    return [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.jpg', '.JPG', '.jpeg', '.png'))]

def calibrate_charuco(images, marker_size, square_size):
    # Calibration function using Charuco markers
    aruco_dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    charuco_board = cv2.aruco.CharucoBoard(BOARD_SIZE, square_size, marker_size, aruco_dictionary)
    aruco_params = cv2.aruco.DetectorParameters()

    corners_list, ids_list = [], []

    for img_file in images:
        # Load each image, convert to grayscale, and detect markers
        image = cv2.imread(img_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray_image,
            aruco_dictionary,
            parameters=aruco_params
        )

        # Interpolate Charuco corners from detected markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray_image,
            board=charuco_board
        )

        # If enough Charuco corners are detected, add to the lists
        if response > 20:
            corners_list.append(charuco_corners)
            ids_list.append(charuco_ids)

    # Calibrate the camera using Charuco corners
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list,
        charucoIds=ids_list,
        board=charuco_board,
        imageSize=gray_image.shape,
        cameraMatrix=None,
        distCoeffs=None)

    return [ret, mtx, dist, rvecs, tvecs]

if __name__ == "__main__":
    # Main execution when the script is run
    image_files = get_image_files(images_folder)

    if not image_files:
        print("No images found in the folder.")
    else:
        print(f'Number of images read: {len(image_files)}')
        # Perform calibration and retrieve results
        results = calibrate_charuco(image_files, MARKER_SIZE, SQUARE_SIZE)
        ret, mtx, dist, rvecs, tvecs = results

        # Print calibration results
        print('\nCalibration Results:')
        print('---------------------')
        print('Calibration Matrix:')
        print(mtx)
        print('\nRadial Distortion Coefficients:')
        print(dist)
        print('\nReturn Value:')
        print(ret)
