from PIL import Image
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# مسار المجلدات
data_dir = "C:/Users/xsale/OneDrive/سطح المكتب/face mask/dataset"
categories = ["with_mask", "without_mask"]

# إعداد البيانات
data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)  # 0 لـ with_mask و 1 لـ without_mask

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            img = Image.open(img_path).convert('RGB')  # فتح الصورة وتحويلها إلى RGB
            resized_img = img.resize((128, 128))  # تغيير حجم الصورة إلى 128x128
            data.append(np.array(resized_img))
            labels.append(class_num)
        except Exception as e:
            print(f"Warning: Unable to load image {img_path}. Error: {e}")

# تحويل القائمة لمصفوفة
data = np.array(data) / 255.0  # عادي بين 0 و 1

# تحويل التسميات إلى فئات (One-Hot Encoding)
if len(labels) > 0:
    labels = to_categorical(np.array(labels))
else:
    print("No labels available. Exiting.")
    exit(1)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# زيادة البيانات
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
)

# إنشاء نموذج الشبكة العصبية المحسن
model = Sequential()

# الطبقة الأولى: طبقة تلافيف
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# الطبقة الثانية: طبقة تلافيف
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# الطبقة الثالثة: طبقة تلافيف
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# الطبقة الرابعة: طبقة تلافيف
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# تسطيح الطبقات
model.add(Flatten())

# الطبقة الكثيفة: طبقة مخفية
model.add(Dense(512, activation='relu'))  # زيادة عدد الوحدات
model.add(BatchNormalization())
model.add(Dropout(0.5))

# الطبقة الأخيرة: طبقة الخرج
model.add(Dense(2, activation='softmax'))

# تجميع النموذج
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج مع زيادة البيانات
epochs = 70  # زيادة عدد الدورات التدريبية هنا
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=epochs)

# تقييم النموذج
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# حفظ النموذج
model.save('face_mask_detector_model_improved.h5')
print("Model saved successfully!")
