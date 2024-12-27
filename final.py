import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Load the trained model
model = load_model('C:/Users/xsale/OneDrive/سطح المكتب/face mask/model.h5')

# Prepare the folder to save images
output_folder = "No_Mask_Screenshots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Global variables
cap = None
photo_count = 0
max_photos = 3

# Function to prepare the image
def prepare_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start camera function
def start_camera():
    global cap, photo_count
    photo_count = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open the camera")
        return
    display_camera()

# Display camera and process frames
def display_camera():
    global cap, photo_count
    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    # Prepare the image for prediction
    img_array = prepare_image(frame)
    prediction = model.predict(img_array)
    mask_probability = prediction[0][0] * 100
    predicted_class = np.argmax(prediction, axis=1)

    # Display text on the frame
    if predicted_class[0] == 0:
        text = f"Mask Detected: {mask_probability:.2f}%"
        color = (0, 255, 0)
    else:
        text = f"No Mask Detected: {100 - mask_probability:.2f}%"
        color = (0, 0, 255)

        # Save the frame if "No Mask" is detected
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(output_folder, f"No_Mask_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        photo_count += 1
        status_label.config(text=f"Captured image #{photo_count} of someone without a mask")

        # Temporary message display
        if photo_count <= max_photos:
            top = tk.Toplevel()
            top.title("Capture Notification")
            tk.Label(top, text="Captured an image of someone without a mask.").pack(padx=20, pady=20)
            top.after(1000, top.destroy)  # Close the message automatically after 1 second

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame in the window
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img_tk = ImageTk.PhotoImage(img)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    # Call the function again
    camera_label.after(10, display_camera)

# Stop camera function
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        camera_label.config(image="")
        status_label.config(text="Camera stopped")

# Function to enlarge the photo
def enlarge_photo(filepath):
    top = tk.Toplevel()
    top.title("Enlarge Photo")
    img = Image.open(filepath)
    img_tk = ImageTk.PhotoImage(img)
    lbl = tk.Label(top, image=img_tk)
    lbl.image = img_tk
    lbl.pack()

# Function to view saved photos
def open_saved_photos():
    photos = os.listdir(output_folder)
    if not photos:
        messagebox.showinfo("No Photos", "No saved photos to display.")
        return

    # Create a new window to display photos
    top = tk.Toplevel()
    top.title("Saved Photos")
    top.geometry("500x500")
    top.configure(bg="#f0f8ff")

    def delete_photo(photo_name):
        if photo_name:
            filepath = os.path.join(output_folder, photo_name)
            if os.path.exists(filepath):
                os.remove(filepath)
                messagebox.showinfo("Deleted", f"Deleted photo: {photo_name}")
                top.destroy()
                open_saved_photos()

    def delete_all_photos():
        for photo in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, photo))
        messagebox.showinfo("Deleted", "All photos have been deleted.")
        top.destroy()
        open_saved_photos()

    delete_all_btn = tk.Button(top, text="Delete All Photos", bg="#ff6f61", fg="white", command=delete_all_photos)
    delete_all_btn.pack(pady=10)

    for photo in photos:
        filepath = os.path.join(output_folder, photo)
        img = Image.open(filepath)
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        frame = tk.Frame(top, bg="#f0f8ff")
        frame.pack(pady=5)

        lbl = tk.Label(frame, image=img_tk)
        lbl.image = img_tk
        lbl.pack(side=tk.LEFT, padx=10)

        btn_delete = tk.Button(frame, text="Delete", bg="#ff6f61", fg="white",
                               command=lambda p=photo: delete_photo(p))
        btn_delete.pack(side=tk.LEFT, padx=10)

        btn_enlarge = tk.Button(frame, text="Enlarge", bg="#61a1ff", fg="white",
                                command=lambda p=filepath: enlarge_photo(p))
        btn_enlarge.pack(side=tk.LEFT, padx=10)

# Create the user interface
root = tk.Tk()
root.title("Face Mask Detection System")
root.geometry("1920x1080")  # Full screen
root.configure(bg="#d1f7ff")

# Add university logo
logo_path = "C:/Users/xsale/OneDrive/سطح المكتب/77410993f197ab8d4fba65d6c7fe1327.png"
logo_img = Image.open(logo_path).resize((150, 150), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(root, image=logo_tk, bg="#d1f7ff")
logo_label.place(x=20, y=20)

# Camera frame
camera_frame = ttk.LabelFrame(root, text="Camera", style="Custom.TLabelframe")
camera_frame.place(relx=0.5, rely=0.4, anchor=tk.CENTER, width=1200, height=800)

camera_label = tk.Label(camera_frame, bg="#000")
camera_label.pack(fill=tk.BOTH, expand=True)

# Button frame
button_frame = ttk.Frame(root)
button_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

btn_start = ttk.Button(button_frame, text="Start Camera", command=start_camera)
btn_start.grid(row=0, column=0, padx=10)

btn_stop = ttk.Button(button_frame, text="Stop Camera", command=stop_camera)
btn_stop.grid(row=0, column=1, padx=10)

btn_view = ttk.Button(button_frame, text="View Saved Photos", command=open_saved_photos)
btn_view.grid(row=0, column=2, padx=10)

# Status frame
status_frame = ttk.LabelFrame(root, text="Status", style="Custom.TLabelframe")
status_frame.place(relx=0.5, rely=0.95, anchor=tk.CENTER, width=1200, height=50)

status_label = tk.Label(status_frame, text="Press 'Start Camera' to begin", anchor="w", bg="#d1f7ff", font=("Arial", 12))
status_label.pack(side=tk.LEFT, padx=10)

# Customize styles
style = ttk.Style()
style.configure("Custom.TLabelframe", background="#d1f7ff", font=("Arial", 14, "bold"))
style.configure("TButton", font=("Arial", 12), padding=5)

# Run the application
root.mainloop()
