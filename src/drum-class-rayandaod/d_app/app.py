import sys
import os
import math
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QListWidgetItem
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

sys.path.append(os.path.abspath(os.path.join('')))

from config import *
from z_helpers.paths import *
from c_train.predict import predict

WIN_WIDTH = 1200
WIN_HEIGHT = 600

here = Path(__file__).parent
style_sheet = open(here / APP_STYLE_SHEET)

DATASET_FOLDER = '20210609-025547-My_samples'
MODEL_FOLDER = 'RF_20210609-234422'


def addSample(table, sample_path):
    row = table.rowCount()
    table.setRowCount(row + 1)
    cell = QTableWidgetItem(sample_path)
    cell.setFlags(Qt.ItemIsEnabled)
    table.setItem(row, 0, cell)


def fill_predictions(table, predictions_dict, unreadable_files, quiet_outliers):
    for i, pred in predictions_dict.items():
        cell = QTableWidgetItem(pred)
        cell.setFlags(Qt.ItemIsEnabled)
        table.setItem(int(i), 1, cell)

    for i, pred in unreadable_files.items():
        cell = QTableWidgetItem('Unreadable')
        cell.setFlags(Qt.ItemIsEnabled)
        table.setItem(int(i), 1, cell)

    for i, pred in quiet_outliers.items():
        cell = QTableWidgetItem('Too quiet')
        cell.setFlags(Qt.ItemIsEnabled)
        table.setItem(int(i), 1, cell)


class TableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Sample', 'Class'])

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

            for url in event.mimeData().urls():
                if url.isLocalFile():
                    if os.path.isdir(url.path()):
                        for input_file in Path(url.path()).glob('**/*.wav'):
                            absolute_path_name = input_file.resolve().as_posix()
                            sample_path = absolute_path_name.replace(GlobalConfig.SAMPLE_LIBRARY, '')
                            addSample(self, sample_path)
                    elif os.path.isfile(url.path()):
                        absolute_path_name = Path(url.path()).resolve().as_posix()
                        sample_path = absolute_path_name.replace(GlobalConfig.SAMPLE_LIBRARY, '')
                        addSample(self, sample_path)
        else:
            event.ignore()

    def getSamplePathDict(self):
        sample_path_dict = dict()
        for i in range(self.rowCount()):
            sample_path_dict[str(i)] = Path(GlobalConfig.SAMPLE_LIBRARY + self.item(i, 0).text())
        return sample_path_dict


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drum Classification')
        self.setFixedSize(WIN_WIDTH, WIN_HEIGHT)

        layout = QVBoxLayout()

        table = TableWidget()
        layout.addWidget(table)
        table.setFixedSize(1160, 500)
        table.setColumnWidth(0, math.ceil(table.width() / 4 * 3))
        table.setColumnWidth(1, math.floor(table.width() / 4))

        self.btn = QPushButton('Predict', self)
        self.btn.clicked.connect(lambda: self.onPressButton(table))
        layout.addWidget(self.btn)

        self.setLayout(layout)
        self.show()

    def onPressButton(self, table):
        sample_path_dict = table.getSamplePathDict()
        # DATASET_FOLDER = self.combo_datasets.currentText()
        # MODEL_FOLDER = self.combo_models.currentText()
        try:
            prediction_dict, unreadable_files, quiet_outliers = predict(sample_path_dict, DATASET_FOLDER, MODEL_FOLDER,
                                                                        is_sample_dict=True)
            fill_predictions(table, prediction_dict, unreadable_files, quiet_outliers)
        except:
            print("Unexpected error:", sys.exc_info())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MainWidget()
    sys.exit(app.exec_())
