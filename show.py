import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from my import emotion_discern

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.upload_button = tk.Button(self, text="上传图片", command=self.upload_img)
        self.upload_button.pack(side="top")

        self.video_button = tk.Button(self, text="打开摄像头", command=self.open_video)
        self.video_button.pack(side="top")

        self.quit = tk.Button(self, text="退出", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def upload_img(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            emotion = emotion_discern()
            emotion_result = emotion.image_discern(file_path)
            self.upload_button.config(text=str(emotion_result))
            img = Image.open(file_path)
            img = ImageTk.PhotoImage(img)
            panel = tk.Label(root, image=img)
            panel.image = img
            panel.pack(side="bottom", fill="both", expand="yes")

    def open_video(self):
        emotion = emotion_discern()
        emotion.video_discern()


root = tk.Tk()
app = Application(master=root)
app.mainloop()
