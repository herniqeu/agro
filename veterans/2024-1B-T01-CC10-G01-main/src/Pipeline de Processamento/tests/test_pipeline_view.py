import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt

from image_process.pipeline import PipelineViewer

app = QApplication(sys.argv)

class TestPipelineViewer(unittest.TestCase):
    def setUp(self):
        self.history = [
            (np.zeros((10, 10, 3)), "Filter Applied", "Filter"),
            (np.zeros((10, 10, 3)), "Cropped", "Crop"),
            (np.zeros((10, 10, 3)), "Augmented", "Augmentation")
        ]
        self.viewer = PipelineViewer(self.history)

    @patch.object(QMainWindow, 'show')
    @patch('PyQt5.QtWidgets.QApplication.__init__', return_value=None)
    @patch('PyQt5.QtWidgets.QApplication.exec', return_value=0)
    def test_plot_history(self, mock_exec, mock_init, mock_show):
        # Test that plot_history setups up the GUI correctly
        self.viewer.plot_history()
        mock_init.assert_called_once()
        mock_exec.assert_called_once()
        mock_show.assert_called_once()

    @patch('PyQt5.QtWidgets.QLabel.setAlignment')
    @patch('PyQt5.QtWidgets.QLabel.setPixmap')
    def test_add_image_to_grid(self, mock_set_pixmap, mock_set_alignment):
        # Test that _add_image_to_grid method sets the pixmap and alignment correctly
        grid_layout = MagicMock()
        self.viewer._add_image_to_grid(self.history[0][0], "Test Image", "Filter", grid_layout, 0, 0)
        mock_set_pixmap.assert_called_once()
        mock_set_alignment.assert_called_once_with(Qt.AlignCenter)

if __name__ == '__main__':
    unittest.main()
