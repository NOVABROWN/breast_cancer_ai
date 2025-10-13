import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_preprocessing import preprocess_tabular
from image_preprocessing import load_images_from_folder

# Load tabular data
X_tab, y = preprocess_tabular("../data/raw/data.csv")

# Load image data
X_img, filenames = load_images_from_folder("../data/raw/images")

# Ensure alignment: images and tabular rows must match
assert X_img.shape[0] == X_tab.shape[0], "Number of images and tabular rows must be the same!"

# Image branch
image_input = Input(shape=(224,224,3))
x = Conv2D(32, (3,3), activation='relu')(image_input)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

# Tabular branch
tabular_input = Input(shape=(X_tab.shape[1],))
y_tab = Dense(64, activation='relu')(tabular_input)
y_tab = Dense(32, activation='relu')(y_tab)

# Fusion
combined = Concatenate()([x, y_tab])
z = Dense(64, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[image_input, tabular_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit([X_img, X_tab], y, epochs=20, batch_size=16, validation_split=0.2)

# Save model
model.save("../models/multimodal_model.h5")
print("💾 Multimodal model saved successfully!")
