# Program to extract img and labels from LISATS and converting them to darknet format.

# Python program for converting the ppm files from The Belgium Traffic Sign dataset to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.
# By Angel Igareta for SaferAuto [https://github.com/angeligareta/SaferAuto]
import csv
from common_config import *

LISATS_ROOT_PATH = "C:/Users/anees/Documents/Research CNN/LISA/"
RESIZE_PERCENTAGE = 0.8
DB_PREFIX = 'lisats-'


COMBINED_ANNOTATIONS_FILE_PATH = LISATS_ROOT_PATH + "allAnnotations.csv"
INPUT_PATH = LISATS_ROOT_PATH + "/"
BACKGROUND_IMG_PATH = LISATS_ROOT_PATH + "input-img-bg/"


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    traffic_sign_classes["0-addedLane"] = ["addedLane"]
    traffic_sign_classes["1-stop"] = ["stop"]
    traffic_sign_classes["2-curveRight"] = ["curveRight"]
    traffic_sign_classes["3-dip"] = ["dip"]
    traffic_sign_classes["4-intersection"] = ["intersection"]
    traffic_sign_classes["5-laneEnds"] = ["laneEnds"]
    traffic_sign_classes["6-merge"] = ["merge"]
    traffic_sign_classes["7-pedestrianCrossing"] = ["pedestrianCrossing"]
    traffic_sign_classes["8-signalAhead"] = ["signalAhead"]
    traffic_sign_classes["9-stopAhead"] = ["stopAhead"]
    traffic_sign_classes["10-thruMergeLeft"] = ["thruMergeLeft"]
    traffic_sign_classes["11-thruMergeRight"] = ["thruMergeRight"]
    traffic_sign_classes["12-turnLeft"] = ["turnLeft"]
    traffic_sign_classes["13-turnRight"] = ["turnRight"]
    traffic_sign_classes["14-yieldAhead"] = ["yieldAhead"]
    traffic_sign_classes["15-doNotPass"] = ["doNotPass"]
    traffic_sign_classes["16-keepRight"] = ["keepRight"]
    traffic_sign_classes["17-rightLaneMustTurn"] = ["rightLaneMustTurn"]
    traffic_sign_classes["18-speedLimit15"] = ["speedLimit15"]
    traffic_sign_classes["19-speedLimit25"] = ["speedLimit25"]
    traffic_sign_classes["20-speedLimit30"] = ["speedLimit30"]
    traffic_sign_classes["21-speedLimit35"] = ["speedLimit35"]
    traffic_sign_classes["22-speedLimit40"] = ["speedLimit40"]
    traffic_sign_classes["23-speedLimit45"] = ["speedLimit45"]
    traffic_sign_classes["24-speedLimit50"] = ["speedLimit50"]
    traffic_sign_classes["25-speedLimit55"] = ["speedLimit55"]
    traffic_sign_classes["26-speedLimit65"] = ["speedLimit65"]
    traffic_sign_classes["27-truckSpeedLimit55"] = ["truckSpeedLimit55"]
    traffic_sign_classes["28-speedLimitUrdbl"] = ["speedLimitUrdbl"]

    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    left_x = float(row[2]) / width_proportion
    bottom_y = float(row[3]) / height_proportion
    right_x = float(row[4]) / width_proportion
    top_y = float(row[5]) / height_proportion

    object_class = row[1]
    # print(object_class)
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if SHOW_IMG:
        show_img(resize_img_plt(input_img, image_width, image_height), left_x, bottom_y, (right_x - left_x), (top_y - bottom_y))

    return parse_darknet_format(object_class_adjusted, image_width, image_height, left_x, bottom_y, right_x, top_y)


def update_global_variables(train_pct, test_pct, valid_pct, color_mode, verbose, false_data, output_img_ext):
    global TRAIN_PROB, TEST_PROB, VALID_PROB, COLOR_MODE, SHOW_IMG, ADD_FALSE_DATA, OUTPUT_IMG_EXTENSION
    TRAIN_PROB = train_pct
    TEST_PROB = test_pct
    VALID_PROB = valid_pct
    COLOR_MODE = color_mode
    SHOW_IMG = verbose
    ADD_FALSE_DATA = false_data
    OUTPUT_IMG_EXTENSION = output_img_ext


def read_dataset(output_train_text_path, output_test_text_path, output_valid_text_path, output_train_dir_path, output_test_dir_path, output_valid_dir_path):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes()
    initialize_classes_counter()

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")
    valid_text_file = open(output_valid_text_path, "a+")

    gt_file = open(COMBINED_ANNOTATIONS_FILE_PATH)  # Annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # CSV parser for annotations file


    # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
    for row in gt_reader:
        filename = row[0].split("/")[-1][:-4]
        file_path = INPUT_PATH + row[0]

        if os.path.isfile(file_path):
            input_img = read_img_plt(file_path)
            darknet_label = calculate_darknet_format(input_img, row)
            object_class_adjusted = int(darknet_label.split()[0])

            # print(object_class_adjusted)

            if filename not in img_labels.keys():  # If it is the first label for that img
                img_labels[filename] = [file_path]

            if object_class_adjusted != OTHER_CLASS:  # Add only useful labels (not false negatives)
                img_labels[filename].append(darknet_label)

    # COUNT FALSE NEGATIVES (IMG WITHOUT LABELS)
    total_false_negatives_dir = {}
    total_annotated_images_dir = {}
    for filename in img_labels.keys():
        img_label_subset = img_labels[filename]
        if len(img_label_subset) == 1:
            total_false_negatives_dir[filename] = img_label_subset
        else:
            total_annotated_images_dir[filename] = img_label_subset

    print('TOTAL ANNOTATED IMAGES: ' + str(len(total_annotated_images_dir.keys())))
    print('TOTAL FALSE NEGATIVES: ' + str(len(total_false_negatives_dir.keys())))

    # SET ANNOTATED IMAGES IN TRAIN OR TEST DIRECTORIES
    # max_imgs = 1000
    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        file_type = rand.choices(["train", "test", "valid"], [TRAIN_PROB, TEST_PROB, VALID_PROB])[0]
        output_filename = DB_PREFIX + filename

        if file_type == "train":
            write_data_updated(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, file_type)
        elif file_type == "test":
            write_data_updated(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, file_type)
        else:
            write_data_updated(output_filename, input_img, input_img_labels, valid_text_file, output_valid_dir_path, file_type)

        # if train_file:
        #     write_data(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, train_file)
        # else:
        #     write_data(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, train_file)

        # max_imgs -= 1
        # if max_imgs == 0:
        #    break

    gt_file.close()
    train_text_file.close()
    test_text_file.close()
    valid_text_file.close()

    return classes_counter_train, classes_counter_test, classes_counter_valid


# read_dataset(OUTPUT_TRAIN_TEXT_PATH, OUTPUT_TEST_TEXT_PATH, OUTPUT_TRAIN_DIR_PATH, OUTPUT_TEST_DIR_PATH)
