import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import f1_score

# Define paths
train_data_dir = '/Users/yanalaraghuvamshireddy/Downloads/test_data'
model_save_path = '/Users/yanalaraghuvamshireddy/Downloads/best_model.keras'

# Parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 10

# Load and preprocess data
train_dataset = image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Preprocess the data
def preprocess(ds):
    return ds.map(lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y))

train_dataset = preprocess(train_dataset)
val_dataset = preprocess(val_dataset)

# Compute class weights
labels = np.concatenate([y for _, y in train_dataset], axis=0)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))

# Define and compile model 1 (MobileNetV3Large)
def create_model_1():
    base_model = MobileNetV3Large(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Define and compile model 2 (InceptionV3)
def create_model_2():
    base_model = InceptionV3(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

model1 = create_model_1()
model2 = create_model_2()

model1.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the models
history1 = model1.fit(train_dataset, validation_data=val_dataset, epochs=epochs, class_weight=class_weights_dict)
history2 = model2.fit(train_dataset, validation_data=val_dataset, epochs=epochs, class_weight=class_weights_dict)

# Define an ensemble model
def create_ensemble_model(model1, model2):
    input_layer = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    output1 = model1(input_layer)
    output2 = model2(input_layer)
    average_output = tf.keras.layers.Average()([output1, output2])
    ensemble_model = Model(inputs=input_layer, outputs=average_output)
    return ensemble_model

ensemble_model = create_ensemble_model(model1, model2)
ensemble_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the ensemble model
def evaluate_ensemble_model(ensemble_model, val_dataset):
    y_true = []
    y_pred = []
    
    for images, labels in val_dataset:
        predictions1 = model1.predict(images)
        predictions2 = model2.predict(images)
        ensemble_predictions = (predictions1 + predictions2) / 2
        y_true.extend(labels.numpy())
        y_pred.extend(ensemble_predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    f1 = f1_score(y_true, y_pred_binary)
    return f1

# Compute F1 Score
f1_score_val = evaluate_ensemble_model(ensemble_model, val_dataset)
print(f"Ensemble F1 Score: {f1_score_val:.4f}")
