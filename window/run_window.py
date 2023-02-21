from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from PyQt5 import QtGui,QtWidgets,QtCore
from window.Ui_untitled import Ui_MainWindow
import sys
from QA.get_res import get_answer
import warnings,torch,re

class MyWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.answer.setReadOnly(True)
        self.submit.clicked.connect(self.submit_handler)
        
        
        
        self.exit.clicked.connect(QtCore.QCoreApplication.quit)
        self.setWindowTitle("ClueAI文字版")
        warnings.filterwarnings("ignore")
        if torch.cuda.is_available:
            infos=str(torch.cuda.get_device_properties(0))
            infos=str(re.findall(r'[(](.*?)[)]', infos))
            info=infos.split(",")
            info=info[3].split("=")
            info=info[1].split("MB")
            info=int(info[0])
        
            if info>6000:
                device_s="cuda"
            else:
                device_s="cpu"
        else: 
            device_s="cpu"
        self.device=device_s

    

    
            

    def submit_handler(self):
        
        
        text=self.question.toPlainText()
        if text=='':
            msg_box = QMessageBox(QMessageBox.Critical, '错误！', '请输入文字')
            msg_box.exec_()
        else:
            
            msg_box = QMessageBox(QMessageBox.Information, '提示！', '使用'+self.device+'进行推理！')
            msg_box.exec_()
            res_txt=get_answer(text,self.device)
            self.answer.setPlainText(res_txt)
            
        
    


            
        


def run():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # application 对象
    app = QApplication(sys.argv)
    
    # QMainWindow对象
    mainwindow = MyWindow()



    # 显示
    mainwindow.show()
    app.exec_()
