import sys
import os
from kivy.app import App
from kivy.core.window import Window

sys.path.append(os.path.abspath(os.path.join('')))

from d_app.predict import predict

DATASET_FOLDER_NAME = '20210609-025547-My_samples'
MODEL_FOLDER_NAME = 'NN_20210609-110954'


def _on_file_drop(window, folder_path):
    print(folder_path.decode("utf-8"))
    drum_types = predict(folder_path.decode("utf-8"), DATASET_FOLDER_NAME, MODEL_FOLDER_NAME)
    print(drum_types)


class WindowFileDropExampleApp(App):
    def build(self):
        Window.bind(on_dropfile=_on_file_drop)
        return


if __name__ == '__main__':

    WindowFileDropExampleApp().run()
