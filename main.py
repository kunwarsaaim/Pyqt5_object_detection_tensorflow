from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QGridLayout, QRadioButton, QDoubleSpinBox, QLabel, QWidget, QFileDialog
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
import os

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util

from models.research.object_detection.utils import visualization_utils as vis_util




class Window(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Object Detection Tensorflow"
        self.top=100
        self.left=100
        self.width=1200
        self.height=900
        self.folder_path = ""
        self.files = []
        self.n =0
        self.n_curr = 0
        self.model = ''
        self.threshold = float(0.5)
        self.filter_class = None
        self.detection_graph =''
        self.category_index = {}
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)

        self.CreateLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.groupboxleft)
        hbox.addWidget(self.image_groupbox)
        hbox.addWidget(self.groupboxright)
        self.setLayout(hbox)
        self.show()

    def CreateLayout(self):
        self.groupboxleft = QGroupBox()
        gridlayoutleft = QGridLayout()

        self.select_folder = QPushButton("Select Folder",self)
        self.select_folder.clicked.connect(self.select_folder_dir)
        self.select_folder.setIconSize(QtCore.QSize(40,40))
        self.select_folder.setMinimumHeight(40)
        gridlayoutleft.addWidget(self.select_folder,0,0)

        self.next_image = QPushButton("Next Image",self)
        self.next_image.clicked.connect(self.next_image_change)
        self.next_image.setIconSize(QtCore.QSize(40,40))
        self.next_image.setMinimumHeight(40)
        gridlayoutleft.addWidget(self.next_image,1,0)

        self.previous_image = QPushButton("Previous Image",self)
        self.previous_image.clicked.connect(self.next_image_change)
        self.previous_image.setIconSize(QtCore.QSize(40,40))
        self.previous_image.setMinimumHeight(40)
        gridlayoutleft.addWidget(self.previous_image,2,0)
        self.groupboxleft.setLayout(gridlayoutleft)


        self.image_groupbox = QWidget()
        vbox_image = QVBoxLayout()
        self.labelImage = QLabel(self)
        self.pixmap = QPixmap('')
        self.labelImage.setPixmap(self.pixmap)
        vbox_image.addWidget(self.labelImage)
        self.image_groupbox.setLayout(vbox_image)
        self.image_groupbox.setMinimumWidth(700)
        


        self.groupboxright = QGroupBox()
        gridlayoutright = QGridLayout()

        self.select_model = QPushButton("Select Model",self)
        self.select_model.clicked.connect(self.model_selection)
        self.select_model.setIconSize(QtCore.QSize(40,40))
        self.select_model.setMinimumHeight(20)
        gridlayoutright.addWidget(self.select_model,0,0)

        #radio buttons for model type
        radio_group_model = QGroupBox()
        qv_groupbox_model_type = QVBoxLayout()
        self.radio_frcnn = QRadioButton("FRCNN")
        self.radio_frcnn.setChecked(True)
        qv_groupbox_model_type.addWidget(self.radio_frcnn)
        self.radio_mobilenet = QRadioButton("Mobile Net")
        qv_groupbox_model_type.addWidget(self.radio_mobilenet)
        self.radio_ssd = QRadioButton("SSD")
        qv_groupbox_model_type.addWidget(self.radio_ssd)
        radio_group_model.setLayout(qv_groupbox_model_type)
        gridlayoutright.addWidget(radio_group_model,1,0)
        #adio buttons for model type end
        
        
        spin_group = QGroupBox()
        qh_groupbox_threshold = QHBoxLayout()
        self.detection_threshold = QPushButton("Detection Threshold")
        self.detection_threshold.clicked.connect(self.threshold_value)
        self.detection_threshold.setIconSize(QtCore.QSize(40,40))
        self.detection_threshold.setMinimumHeight(20)
        qh_groupbox_threshold.addWidget(self.detection_threshold)
        self.doublespinbox = QDoubleSpinBox(self)
        self.doublespinbox.setRange(0,float(1))
        self.doublespinbox.setSingleStep(0.05)
        qh_groupbox_threshold.addWidget(self.doublespinbox)
        spin_group.setLayout(qh_groupbox_threshold)
        gridlayoutright.addWidget(spin_group,2,0)





        #radio buttons for class filter
        radio_group_filter = QGroupBox("Label Filter")
        radio_group_filter.setFont(QtGui.QFont("Sanserif",15))
        qv_groupbox_filter = QVBoxLayout()
        self.radio_person = QRadioButton("Person")
        self.radio_person.toggled.connect(lambda:self.filter(self.radio_person))
        qv_groupbox_filter.addWidget(self.radio_person)
        self.radio_dog = QRadioButton("Dog")
        self.radio_dog.toggled.connect(lambda:self.filter(self.radio_dog))
        qv_groupbox_filter.addWidget(self.radio_dog)
        self.radio_cat = QRadioButton("Cat")
        self.radio_cat.toggled.connect(lambda:self.filter(self.radio_cat))
        qv_groupbox_filter.addWidget(self.radio_cat)
        self.radio_bottle = QRadioButton("Bottle")
        self.radio_bottle.toggled.connect(lambda:self.filter(self.radio_bottle))
        qv_groupbox_filter.addWidget(self.radio_bottle)
        self.radio_chair = QRadioButton("Chair")
        self.radio_chair.toggled.connect(lambda:self.filter(self.radio_chair))
        qv_groupbox_filter.addWidget(self.radio_chair)
        radio_group_filter.setLayout(qv_groupbox_filter)
        gridlayoutright.addWidget(radio_group_filter,3,0)
        #radio buttons for class filter



        self.detect = QPushButton("Detect",self)
        self.detect.clicked.connect(self.predict_fun)
        self.detect.setIconSize(QtCore.QSize(40,40))
        self.detect.setMinimumHeight(20)
        gridlayoutright.addWidget(self.detect,4,0)

        self.groupboxright.setLayout(gridlayoutright)



    def select_folder_dir(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Select Directory")
        print(self.folder_path)
        self.files = os.listdir(self.folder_path)
        print(self.files)
        self.n = len(self.files)
        self.pixmap = QPixmap(os.path.join(self.folder_path,self.files[self.n_curr]))
        # self.pixmap = self.pixmap.scaledToWidth(540)
        # self.pixmap =  self.pixmap.scaledToHeight(640)
        self.labelImage.setPixmap(self.pixmap)

    def next_image_change(self):
        if self.n_curr == self.n - 1:
            self.n_curr = 0
        else:
            self.n_curr = self.n_curr+1
        self.pixmap = QPixmap(os.path.join(self.folder_path,self.files[self.n_curr]))
        # self.pixmap = self.pixmap.scaledToWidth(540)
        # self.pixmap =  self.pixmap.scaledToHeight(640)
        self.labelImage.setPixmap(self.pixmap)

    def previous_image_change(self):
        if self.n_curr == 0:
            self.n_curr = self.n -1
        else:
            self.n_curr = self.n_curr - 1
        self.pixmap = QPixmap(os.path.join(self.folder_path,self.files[self.n_curr]))
        # self.pixmap = self.pixmap.scaledToWidth(540)
        # self.pixmap =  self.pixmap.scaledToHeight(640)
        self.labelImage.setPixmap(self.pixmap)
        
    def model_selection(self):
        if self.radio_frcnn.isChecked() == True:
            self.model = "faster_rcnn_inception_v2_coco_2018_01_28"
        elif self.radio_mobilenet.isChecked() == True:
            self.model = "ssdlite_mobilenet_v2_coco_2018_05_09"
        elif self.radio_ssd.isChecked() == True:
            self.model = "ssd_inception_v2_coco_2018_01_28"
        MODEL_NAME = self.model
        MODEL_FILE = MODEL_NAME + '.tar.gz'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('models','research','object_detection','data', 'mscoco_label_map.pbtxt') 

        tar_file = tarfile.open(os.path.join('downloads',MODEL_FILE))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        self.category_index = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS,use_display_name=True)
        print(self.model)
       
        
    def threshold_value(self):
        self.threshold = float(self.doublespinbox.value())
        print(self.threshold)

    def filter(self,radio):
        if radio.text() == "Person":
            if radio.isChecked() == True:
                self.filter_class = "Person"
            else:
                radio.setChecked(False)
                self.filter_class = None

        if radio.text() == "Dog":
            if radio.isChecked() == True:
                self.filter_class = "Dog"
            else:
                radio.setChecked(False)
                self.filter_class = None

        if radio.text() == "Cat":
            if radio.isChecked() == True:
                self.filter_class = "Cat"
            else:
                radio.setChecked(False)
                self.filter_class = None

        if radio.text() == "Bottle":
            if radio.isChecked() == True:
                self.filter_class = "Bottle"
            else:
                radio.setChecked(False)
                self.filter_class = None
        
        if radio.text() == "Chair":
            if radio.isChecked() == True:
                self.filter_class = "Chair"
            else:
                radio.setChecked(False)
                self.filter_class = None

        print(self.filter_class)

    def predict_fun(self):
        image = Image.open(os.path.join(self.folder_path,self.files[self.n_curr]))
        image_np = self.load_image_into_array(image)

        image_np_expand = np.expand_dims(image_np,axis=0)

        output_dict = self.run_inference_for_single_image(image_np)
        print(self.filter_class)
        if(self.filter_class!=None):
            output_dict['detection_boxes'] = np.squeeze(output_dict['detection_boxes'])
            output_dict['detection_classes'] = np.squeeze(output_dict['detection_classes'])
            output_dict['detection_scores'] = np.squeeze(output_dict['detection_scores'])
            indices=1
            if(self.filter_class=='Person'):
                indices = np.argwhere(output_dict['detection_classes']==1)
            elif(self.filter_class=='Dog'):
                indices = np.argwhere(output_dict['detection_classes']==18)
            
            elif(self.filter_class=='Cat'):
                indices = np.argwhere(output_dict['detection_classes']==17)
            elif(self.filter_class=='Bottle'):
                indices = np.argwhere(output_dict['detection_classes']==44)
            elif(self.filter_class=='Chair'):
                indices = np.argwhere(output_dict['detection_classes']==62)
            print(indices)
            output_dict['detection_boxes'] = np.squeeze(output_dict['detection_boxes'][indices])
            output_dict['detection_classes'] = np.squeeze(output_dict['detection_classes'][indices])
            output_dict['detection_scores'] = np.squeeze(output_dict['detection_scores'][indices])

        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        self.category_index[0],
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=self.threshold,
        line_thickness=8)
        fig = plt.figure(figsize=(7,5),frameon=False)
        ax = plt.Axes(fig,[0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_np,aspect='auto')
        fig.savefig("output.png",bbox_inches='tight',dpi=fig.dpi)
        self.pixmap = QPixmap('output.png')
        self.labelImage.setPixmap(self.pixmap)
        print("done")

    def load_image_into_array(self,image):
        (im_width,im_height) = image.size
        return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)


    def run_inference_for_single_image(self,image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())