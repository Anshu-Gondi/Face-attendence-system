import cv2
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import messagebox
import csv
from datetime import datetime

# Function to create the data directory if it doesn't exist
def create_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Data folder created.")
    else:
        print("Data folder already exists.")

# Function to log attendance
def log_attendance(name, class_, roll_no, section):
    # Create the CSV file if it doesn't exist
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Class", "Roll No", "Section", "Date", "Time"])

    # Get the current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Check if the person is already marked present today
    with open("attendance.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name and row[1] == class_ and row[2] == roll_no and row[3] == section and row[4] == date_str:
                messagebox.showinfo("Info", f"{name} (Roll No: {roll_no}, Section: {section}) is already marked present today.")
                return

    # Append the attendance log
    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, class_, roll_no, section, date_str, time_str])
    
    messagebox.showinfo("Info", f"Attendance for {name} (Roll No: {roll_no}, Section: {section}) marked at {time_str} on {date_str}.")

# Function to capture face data and save it
def capture_faces():
    # Initialize the video capture and face detector
    video = cv2.VideoCapture(0)  # 0 for webcam
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

    face_data = []
    i = 0

    # Get user input for name, class, roll number, and section
    name = name_entry.get()
    class_ = class_entry.get()
    roll_no = roll_no_entry.get()
    section = section_entry.get()

    # Validate inputs
    if not name or not class_ or not roll_no or not section:
        messagebox.showwarning("Input Error", "Please fill in all fields (Name, Class, Roll No, and Section).")
        return

    # Create a personal folder inside the data folder for each user
    user_folder = f"data/{name}_{class_}_{roll_no}_{section}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Main loop to capture video frames
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = facedetect.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around detected faces and capture face data
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]  # Crop face from the frame
            resized_img = cv2.resize(crop_img, (50, 50))  # Resize to 50x50 pixels

            # Collect face data every 10th frame until we have 100 images
            if len(face_data) < 100 and i % 10 == 0:
                face_data.append(resized_img)
                cv2.putText(frame, f"Captured: {len(face_data)}", org=(50, 50), 
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 255), thickness=2)

                # Save each captured face as an image in the user's folder
                cv2.imwrite(f"{user_folder}/{name}_{len(face_data)}.jpg", resized_img)

        i += 1  # Increment frame counter

        # Display the frame with detected faces
        cv2.imshow("Capturing Faces", frame)

        # Break the loop if 'q' is pressed or we have 100 face samples
        if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 100:
            break

    # Release the video capture and destroy all windows
    video.release()
    cv2.destroyAllWindows()

    # Save face data as a pickle file
    with open(f"{user_folder}/{name}_face_data.pkl", "wb") as f:
        pickle.dump(face_data, f)

    print(f"Face data for {name} saved successfully!")
    messagebox.showinfo("Info", f"Face data for {name} saved successfully in the 'data' folder!")

    # Log attendance
    log_attendance(name, class_, roll_no, section)

# Setup the GUI using tkinter
root = tk.Tk()
root.title("Face Data Capture")
root.geometry("400x300")

# Create a label and input box for user name
tk.Label(root, text="Enter your name:").pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

# Create a label and input box for class
tk.Label(root, text="Enter your class:").pack(pady=5)
class_entry = tk.Entry(root)
class_entry.pack(pady=5)

# Create a label and input box for roll number
tk.Label(root, text="Enter your roll number:").pack(pady=5)
roll_no_entry = tk.Entry(root)
roll_no_entry.pack(pady=5)

# Create a label and input box for section
tk.Label(root, text="Enter your section:").pack(pady=5)
section_entry = tk.Entry(root)
section_entry.pack(pady=5)

# Create a button to start face capture
capture_button = tk.Button(root, text="Start Capture", command=capture_faces)
capture_button.pack(pady=20)

# Create the data folder on startup
create_data_folder()

# Start the GUI loop
root.mainloop()
