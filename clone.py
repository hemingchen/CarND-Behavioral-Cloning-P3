import csv
import datetime
import os
from shutil import copyfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

STEERING_CORRECTION = 0.2
N_EPOCHS = 3
IMAGE_SCALING_FACTOR = 0.5
ADD_FLIPPED_IMAGES = True
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_CHANNEL = 3
DEFAULT_IMAGE_SIZE = (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_CHANNEL)
IMAGE_CROP_TOP = 70
IMAGE_CROP_BOTTOM = 25
IMAGE_CROP_LEFT = 0
IMAGE_CROP_RIGHT = 0
CROPPED_IMAGE_SIZE = (
    DEFAULT_IMAGE_HEIGHT - IMAGE_CROP_TOP - IMAGE_CROP_BOTTOM,
    DEFAULT_IMAGE_WIDTH - IMAGE_CROP_LEFT - IMAGE_CROP_RIGHT,
    DEFAULT_IMAGE_CHANNEL)
SCALED_IMAGE_WIDTH = int(IMAGE_SCALING_FACTOR * DEFAULT_IMAGE_WIDTH)
SCALED_IMAGE_HEIGHT = int(IMAGE_SCALING_FACTOR * DEFAULT_IMAGE_HEIGHT)
DRIVING_LOG_CSV_FILE_NAME = 'driving_log.csv'
ROOT_DATA_FOLDER = './data'
ROOT_TESTED_MODELS_FOLDER = './tested_models'
SELECTED_DATA_SETS = [
    # 'curve_near_beginning_1',
    # 'curve_near_beginning_2',
    # 'recovery_driving',
    'reversed_center_lane_driving_1',
    # 'curve_after_bridge_1',
    # 'curve_after_bridge_2',
    'center_lane_driving_1',
    'center_lane_driving_2',
    'center_lane_driving_3',
    'center_lane_driving_4',
    'center_lane_driving_5',
    'center_lane_driving_6',
    'udacity']
MODEL_META_FILE_NAME = "meta.txt"


def read_csv_lines(file_path, img_path_prefix=None):
    lines = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        csv_header = next(reader)
        for line in reader:
            if img_path_prefix is None:
                lines.append(line)
            else:
                for i in range(0, 3):
                    # Clean up the image path since data was recorded on Windows/macOS
                    line[i] = img_path_prefix + "IMG/" + line[i].strip().replace("\\", "/").split("/")[-1]
                lines.append(line)
    return lines


# sample = line here
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Read in images from center, left and right cameras
                center_image = cv2.imread(ROOT_DATA_FOLDER + '/' + batch_sample[0])
                left_image = cv2.imread(ROOT_DATA_FOLDER + '/' + batch_sample[1])
                right_image = cv2.imread(ROOT_DATA_FOLDER + '/' + batch_sample[2])

                # Convert to RGB since the drive.py sends in RGB images when testing
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                # Read steering and populate for left and right camera images
                center_image_steering_measurement = float(batch_sample[3])
                left_image_steering_measurement = center_image_steering_measurement + STEERING_CORRECTION
                right_image_steering_measurement = center_image_steering_measurement - STEERING_CORRECTION

                images.extend([
                    center_image,
                    left_image,
                    right_image])
                measurements.extend([
                    center_image_steering_measurement,
                    left_image_steering_measurement,
                    right_image_steering_measurement])

                if ADD_FLIPPED_IMAGES:
                    # add flipped images and measurements
                    center_image_flipped = np.fliplr(center_image)
                    left_image_flipped = np.fliplr(left_image)
                    right_image_flipped = np.fliplr(right_image)

                    center_image_steering_measurement_flipped = -center_image_steering_measurement
                    left_image_flipped_steering_measurement = -left_image_steering_measurement
                    right_image_flipped_steering_measurement_flipped = -right_image_steering_measurement

                    images.extend([
                        center_image_flipped,
                        left_image_flipped,
                        right_image_flipped])
                    measurements.extend([
                        center_image_steering_measurement_flipped,
                        left_image_flipped_steering_measurement,
                        right_image_flipped_steering_measurement_flipped])

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


lines = []
for selected_data_set in SELECTED_DATA_SETS:
    lines.extend(read_csv_lines(
        file_path=ROOT_DATA_FOLDER + '/' + selected_data_set + '/' + DRIVING_LOG_CSV_FILE_NAME,
        img_path_prefix=selected_data_set + '/'))

print('\ntotal training set size: {}\n'.format(len(lines)))

# Split all lines into train and validation set
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Get the generators for each set
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Use Nvidia's CNN for autonomous driving
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=DEFAULT_IMAGE_SIZE))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# Train
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=N_EPOCHS)

# Save model
model.save('model.h5')

# Archive model
timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_folder = './' + ROOT_TESTED_MODELS_FOLDER + '/' + timestamp
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
    copyfile('model.h5', model_folder + '/' + 'model.h5')
    with open(model_folder + '/' + MODEL_META_FILE_NAME, 'a') as f:
        f.write('selected_data_set:\n')
        for data_set in SELECTED_DATA_SETS:
            f.write('  ' + data_set + '\n')
        f.write('n_epochs: ' + str(N_EPOCHS) + '\n')
        f.write('add_flipped_images: ' + str(ADD_FLIPPED_IMAGES) + '\n')

# Plot result
# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(model_folder + '/' + 'train_valid_loss.png')
plt.show()
