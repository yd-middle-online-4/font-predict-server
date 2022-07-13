import os
import pickle
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app, expose_headers="Authorization")

uploads_dir = os.path.join(app.instance_path, "uploads")
os.makedirs(uploads_dir, exist_ok=True)


def load_model():
    # BaseLine model
    model = tf.keras.models.Sequential(
        [
            # Note the input shape is the desired size of the image 300x300 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.ZeroPadding2D(padding=1, input_shape=(64, 160, 1)),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.Conv2D(128, (3, 3)),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.ZeroPadding2D(padding=1),
            tf.keras.layers.Conv2D(256, (3, 3)),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # 1024 neuron hidden layer
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation("relu"),
            # 1024 neuron hidden layer
            tf.keras.layers.Dense(256),
            tf.keras.layers.Activation("relu"),
            # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
            tf.keras.layers.Dense(120, activation="softmax"),
        ]
    )
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_weights(model):
    saved_model_path = "instance/saved_models/cnn_weights_v3.hdf5"
    model.load_weights(saved_model_path)
    return model


def load_labels():
    with open("instance/font_labels.bin", "rb") as f:
        folder_maps = pickle.load(f)  # 단 한줄씩 읽어옴
    labels = list(folder_maps.keys())
    return labels


def load_image():
    img_path = "instance/uploads/image.png"
    img = image.load_img(img_path, color_mode="grayscale", target_size=(64, 160))
    x = image.img_to_array(img)
    x = x.reshape(1, 64, 160, 1)
    return x / 255


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"ok": False, "message": "not exist image"})
        file = request.files["image"]
        file.save(os.path.join(uploads_dir, file.filename))
        img = load_image()
        model = load_model()
        model = load_weights(model)
        labels = load_labels()
        pred = model.predict(img)
        idx = np.argmax(pred)
        result = labels[idx]
        return jsonify(
            {
                "ok": True,
                "message": "done",
                "data": {"font": result, "font_idx": f"{idx:04d}"},
            }
        )


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5555, debug=True)
    app.run()
