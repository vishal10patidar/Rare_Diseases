import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

app = Flask(__name__)

class ImageClassifier:
    def __init__(self, train_dir, val_dir, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model = self.create_model()
        self.train_data, self.val_data = self.prepare_data()

    def prepare_data(self):
        # Initialize ImageDataGenerator for training and validation
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Create training data generator
        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        # Create validation data generator
        val_data = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_data, val_data

    def create_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.img_size[0], self.img_size[1], 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def generate_images(self, sample_img_path, num_images):
        sample_img = self.load_image(sample_img_path)
        data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        generated_images = []
        for i in range(num_images):
            img_iterator = data_gen.flow(sample_img)
            augmented_img = next(img_iterator)[0].astype(np.float32)
            generated_images.append(augmented_img)

        return generated_images

classifier = ImageClassifier('C:/Users/Prashant/Desktop/Project/Rare_Diseases/final_yr/Split_smol/Dataset/train', 'C:/Users/Prashant/Desktop/Project/Rare_Diseases/final_yr/Split_smol/Dataset/val') # Update paths here

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_images = int(request.form['num_images'])
        sample_image_path = 'C:/Users/Prashant/Desktop/Project/Rare_Diseases/final_yr/Split_smol/Dataset/val/benign/Copy of ISIC_0014609_downsampled.jpg'  # Update with a valid sample image path

        # Generate images
        generated_images = classifier.generate_images(sample_image_path, num_images)

        # Save generated images to a folder
        os.makedirs('static/generated', exist_ok=True)
        for i, img in enumerate(generated_images):
            plt.imsave(f'static/generated/generated_{i}.png', img)

        return render_template('index.html', generated_images=[f'generated_{i}.png' for i in range(num_images)])

    return render_template('index.html', generated_images=[])

if __name__ == "__main__":
    app.run(debug=True)
