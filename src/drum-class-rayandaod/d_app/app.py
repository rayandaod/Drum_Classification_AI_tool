import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout, \
    QComboBox
from PyQt5.QtCore import Qt

sys.path.append(os.path.abspath(os.path.join('')))

from c_train.predict import predict
from z_helpers.paths import *

here = Path(__file__).parent

DATASET_FOLDER = '20210609-025547-My_samples'
MODEL_FOLDER = 'RF_20210609-234422'


class ListBoxWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.resize(600, 600)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.combo_datasets = QComboBox()
        self.combo_datasets.addItems(next(os.walk(MODELS_PATH))[1])
        self.combo_datasets.currentIndexChanged.connect(self.datasetSelectionChange)
        layout.addWidget(self.combo_datasets)

        self.combo_models = QComboBox()
        self.combo_models.addItems(next(os.walk(MODELS_PATH / self.combo_datasets.currentText()))[1])
        self.combo_models.currentIndexChanged.connect(self.modelSelectionChange)
        layout.addWidget(self.combo_models)

    def datasetSelectionChange(self, i):
        DATASET_FOLDER = self.combo_datasets.currentText()
        for i in range(self.combo_models.count()):
            self.combo_models.removeItem(i)
        self.combo_models.addItems(next(os.walk(MODELS_PATH / DATASET_FOLDER))[1])

    def modelSelectionChange(self, i):
        return

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

            links = []
            for url in event.mimeData().urls():
                # https://doc.qt.io/qt-5/qurl.html
                if url.isLocalFile():
                    links.append(str(url.toLocalFile()))
                else:
                    links.append(str(url.toString()))
            self.addItems(links)
        else:
            event.ignore()


class DrumClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 600)
        style_sheet = open(here / APP_STYLE_SHEET)

        self.listbox_view = ListBoxWidget(self)

        self.btn = QPushButton('Predict', self)
        self.btn.setGeometry(995, 545, 200, 50)
        self.btn.setStyleSheet(style_sheet.read())
        self.btn.clicked.connect(lambda: self.onPressButton())

    def onPressButton(self):
        item = QListWidgetItem(self.listbox_view.currentItem())
        drum_types = predict(item.text(), DATASET_FOLDER, MODEL_FOLDER)
        print(drum_types)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = DrumClassificationWindow()
    window.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')
