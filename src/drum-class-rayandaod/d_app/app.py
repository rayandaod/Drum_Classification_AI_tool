import sys
import os
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

sys.path.append(os.path.abspath(os.path.join('')))

from c_train.predict import predict
from z_helpers.paths import *
from config import *

here = Path(__file__).parent

WIN_WIDTH = 1200
WIN_HEIGHT = 600

style_sheet = open(here / APP_STYLE_SHEET)


class TableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        layout = QHBoxLayout()
        self.setLayout(layout)

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
                    print(url.path())
                    for input_file in Path(url.path()).glob('**/*.wav'):
                        absolute_path_name = input_file.resolve().as_posix()
                        links.append(absolute_path_name.replace(GlobalConfig.SAMPLE_LIBRARY, ''))
                else:
                    links.append(str(url.toString()))
            self.addItems(links)
        else:
            event.ignore()


class ToolBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(WIN_WIDTH, 120)
        layout = QVBoxLayout()

        self.combo_datasets = QComboBox()
        self.combo_datasets.addItems(next(os.walk(MODELS_PATH))[1])
        self.combo_datasets.currentIndexChanged.connect(self.datasetSelectionChange)
        layout.addWidget(self.combo_datasets)

        self.combo_models = QComboBox()
        self.combo_models.addItems(next(os.walk(MODELS_PATH / self.combo_datasets.currentText()))[1])
        self.combo_models.currentIndexChanged.connect(self.modelSelectionChange)
        layout.addWidget(self.combo_models)

        # self.btn = QPushButton('Predict', self)
        # self.btn.setGeometry(0, 545, 200, 50)
        # self.btn.setStyleSheet(style_sheet.read())
        # self.btn.clicked.connect(lambda: self.onPressButton())
        # layout.addWidget(self.btn)

    def datasetSelectionChange(self, i):
        current_dataset = self.combo_datasets.currentText()
        for i in range(self.combo_models.count()):
            self.combo_models.removeItem(i)
        self.combo_models.addItems(next(os.walk(MODELS_PATH / current_dataset))[1])

    def modelSelectionChange(self, i):
        return

    def onPressButton(self):
        item = QListWidgetItem(self.listbox_view.currentItem())
        DATASET_FOLDER = self.combo_datasets.currentText()
        MODEL_FOLDER = self.combo_models.currentText()
        drum_types = predict(item.text(), DATASET_FOLDER, MODEL_FOLDER)
        print(drum_types)


class DrumClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(WIN_WIDTH, WIN_HEIGHT)

        self.table_view = TableWidget(self)
        self.table_view.setGeometry(0, 0, WIN_WIDTH, int(WIN_HEIGHT * 4 / 5))

        self.toolbar_view = ToolBarWidget(self)
        toolbar_height = math.floor(WIN_HEIGHT - self.table_view.height())
        self.toolbar_view.setGeometry(0, WIN_HEIGHT - toolbar_height, WIN_WIDTH, toolbar_height)
        self.toolbar_view.setStyleSheet('background: white')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrumClassificationWindow()
    window.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')
