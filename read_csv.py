import csv

def pixel_to_world_coords(w, h, fx, fy, cx, cy, distance):
    """
    Convert pixel coordinates to real-world coordinates using the pinhole camera model.

    Parameters:
    - w, h: Pixel coordinates (width and height).
    - fx, fy: Focal lengths in pixel units (derived from actual focal length and sensor size).
    - cx, cy: Principal point (usually the center of the image).
    - distance: Distance from the camera to the object in the Z-direction (depth).

    Returns:
    - X, Y, Z: Real-world coordinates.
    """
    X = (w - cx) * distance/fx
    Y = (h - cy) * distance/fy
    Z = distance
    return X, Y, Z


def read_csv(file_path):
    """
    Reads a CSV file and returns the pixel coordinates as a list of tuples with integers.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - corners: A list of tuples, each containing integer (w, h) values.
    """
    corners = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert each value in the row to an integer and append as a tuple
            corners.append((int(row[0]), int(row[1])))
    
    return corners

def get_world_coords():
    # Example usage
    file_path = 'docker/competition_output/18_corners.csv'  # Replace with your CSV file path
    corners = read_csv(file_path)

    # Print the list of tuples with the (w, h) values
    for i, (w, h) in enumerate(corners, start=1):
        print(f"Corner {i}: w = {w}, h = {h}")

    # Camera parameters
    image_width = 1920
    image_height = 1080
    focal_length_mm = 2.3  # Kinect depth camera focal length in mm // 1.8
    sensor_width_mm = 5.120  # Example sensor width in mm // 3.5
    sensor_height_mm = 3.132  # Example sensor height in mm // 2.0

    # Convert focal length to pixels
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = (focal_length_mm / sensor_height_mm) * image_height

    # Assume the principal point is at the center of the image
    cx = image_width / 2
    cy = image_height / 2

    # Distance from camera to cloth in mm
    distance_mm = 1150  # 115 cm

    # Example corners of the cloth in pixel coordinates (w1, h1), (w2, h2), etc.
    #corners_pixels = [
    #    (w3, h3), (w2, h2), (w1, h1), (w4, h4)
    #]

    # Calculate real-world coordinates
    corners_world = []
    for (w, h) in corners:
        X, Y, Z = pixel_to_world_coords(w, h, fx, fy, cx, cy, distance_mm)
        corners_world.append((X, Y, Z))

    # for i in corners_world:
    #     print("Real-world coordinates:", i) s
    #     #print("x: ", i[0], "y: ", i[1], "z: ", i[2])
    print("p1: ", corners_world[2])
    print("p2: ", corners_world[1])
    print("p3: ", corners_world[0])
    print("p4: ", corners_world[3])

    return corners_world

get_world_coords()