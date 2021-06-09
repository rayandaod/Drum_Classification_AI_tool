import sys
import os
from kivy.app import App
from kivy.core.window import Window

sys.path.append(os.path.abspath(os.path.join('')))

from d_app.predict import predict


def _on_file_drop(window, folder_path):
    drum_types = predict(folder_path=folder_path, dataset_folder_name='20210609-025547-My_samples',
                         model_folder_name='NN_20210609-110954')
    return drum_types


class WindowFileDropExampleApp(App):
    def build(self):
        Window.bind(on_dropfile=_on_file_drop)
        return


if __name__ == '__main__':
    WindowFileDropExampleApp().run()
