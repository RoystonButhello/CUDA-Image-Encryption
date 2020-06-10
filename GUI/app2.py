from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QAction,QFileDialog,QWidget	
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
import os 
import sys	
import subprocess
import basics
import cv2																					

class MyWindow(QMainWindow,QWidget):
	def __init__(self):
		super(MyWindow,self).__init__()
		self.setGeometry(500,500,600,600)
		self.initUI()
	
	def initUI(self):
		self.label = QtWidgets.QLabel(self)
		self.label.setText("")
		self.label.setGeometry(250,250,100,100)

		# Select Image button
		self.selectImageButton = QtWidgets.QPushButton(self)
		self.selectImageButton.setText("Select and Run")
		self.selectImageButton.move(250,400)
		self.selectImageButton.clicked.connect(self.getImageFile)

		# Create exit action
		exitAction = QAction(QIcon('exit.png'), '&Exit', self)
		exitAction.setShortcut('Alt+F4')
		exitAction.setStatusTip('Exit Application')
		exitAction.triggered.connect(self.exitCall)

		# Menu bar
		menuBar = self.menuBar()
		menuBar.setNativeMenuBar(False)
		fileMenu = menuBar.addMenu('&File')
		fileMenu.addAction(exitAction)

	# Method for Select Image button
	"""def clicked(self):
		self.label.setText("Button Clicked")
		self.update()

	# Method for label
	def update(self):
		self.label.adjustSize()"""

	def exitCall(self):
		print("\nExited Application")
		QApplication.quit()

	# Connected to Select Image button and openAction
	def getImageFile(self):
		imageName = ""
		imagePath = QFileDialog.getOpenFileName(self, 'Select Image', '/home/saswat/',"Image files (*.png)")
		runPath = QFileDialog.getOpenFileName(self, 'Select run.sh', '/home/saswat/',"linux shell scripts (*.sh)")
		runDirectory = os.path.dirname(runPath[0])
		os.chdir(runDirectory)
		#print(os.getcwd())
		imageName = basics.getFileNameFromPath(imagePath[0])
		runName = basics.getFileNameFromPath(runPath[0])
		image = cv2.imread(imagePath[0])
		cv2.imshow(imageName,image)
		runCommand = "bash " + runName + " " + imagePath[0]
		exitCommand = "exit"
		subprocess.call(runCommand,shell = True)
		print("\nrunCommand = " + runCommand)

def window():
	app = QApplication(sys.argv)
	win = MyWindow()
	win.show()
	subprocess.call("exit",shell = True)
	sys.exit(app.exec_())

window()

