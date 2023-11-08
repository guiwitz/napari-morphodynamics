"""
This module implements a napari widget to create microfilm images interactively
by capturing views.
"""

import pickle
from itertools import cycle
from pathlib import Path
from qtpy.QtWidgets import (QWidget, QPushButton, QSpinBox, QDoubleSpinBox,
QVBoxLayout, QLabel, QComboBox, QCheckBox, QGridLayout,QGroupBox,
QListWidget, QFileDialog, QScrollArea, QAbstractItemView)
from qtpy.QtCore import Qt
from magicgui.widgets import create_widget

import numpy as np
import napari
import skimage

from dask import delayed
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster

from .folder_list_widget import FolderListWidget
from napari_guitils.gui_structures import VHGroup, TabSet
from .base_plot import DataPlotter

from morphodynamics.store.parameters import Param
from morphodynamics.utils import (dataset_from_param, load_alldata, 
                                  export_results_parameters)
from morphodynamics.analysis_par import (
    analyze_morphodynamics, segment_single_frame, 
    compute_spline_windows, segment_and_track, spline_and_window)
from morphodynamics.windowing import label_windows
from morphodynamics.plots.show_plots import (
            show_correlation_core, show_displacement, show_cumdisplacement,
            show_curvature, show_edge_overview, show_geometry)
from morphodynamics.correlation import correlate_arrays

from napari_convpaint.conv_paint import ConvPaintWidget
from napari_convpaint.conv_paint_utils import Classifier

import matplotlib as mpl

mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['font.size'] = 15


class MorphoWidget(QWidget):
    """
    Implentation of a napari plugin offering an interface to the morphodynamics softwere.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer object.
    """
    
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer

        # create a param object
        self.param = Param(
            seg_algo='cellpose'
        )
        self.analysis_path = None
        self.cluster = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setMinimumWidth(400)

        self.tab_names = [
            'Data', 'Segmentation', 'Windowing', 'Signal', 'Morpho', 'Correlations', 'Dask']
        self.tabs = TabSet(
            self.tab_names,
            tab_layouts=[None, None, None, QGridLayout(), None, QGridLayout(), None]
        )

        self.main_layout.addWidget(self.tabs)

        # add widgets to main tab
        self.data_vgroup = VHGroup('1. Select location of data', orientation='G')
        self.tabs.add_named_tab('Data', self.data_vgroup.gbox)

        # files
        self.qcombobox_data_type = QComboBox()
        self.qcombobox_data_type.addItems(['zarr', 'multipage_tiff', 'tiff_series', 'nparray', 'layers'])
        self.data_vgroup.glayout.addWidget(self.qcombobox_data_type, 0, 0, 1, 1)
        self.btn_select_file_folder = QPushButton("Select data folder")
        self.data_vgroup.glayout.addWidget(self.btn_select_file_folder)
        self.file_list = FolderListWidget(napari_viewer)
        self.data_vgroup.glayout.addWidget(self.file_list)
        self.file_list.setMaximumHeight(100)
        

        # channel selection
        self.segm_channel = QListWidget()
        self.segm_channel.setMaximumHeight(100)
        self.signal_channel = QListWidget()
        self.signal_channel.setMaximumHeight(100)
        self.signal_channel.setSelectionMode(QAbstractItemView.ExtendedSelection)

        channel_group = VHGroup('2. Select channels to use', orientation='G')
        self.tabs.add_named_tab('Data', channel_group.gbox)

        channel_group.glayout.addWidget(QLabel('Segmentation'),0,0)
        channel_group.glayout.addWidget(QLabel('Signal'),0,1)
        channel_group.glayout.addWidget(self.segm_channel,1,0)
        channel_group.glayout.addWidget(self.signal_channel,1,1)

        # load data
        load_group = VHGroup('3. Load and display the dataset', orientation='G')
        self.tabs.add_named_tab('Data', load_group.gbox)
        self.btn_load_data = QPushButton("Load")
        load_group.glayout.addWidget(self.btn_load_data)

        # select saving place
        analysis_vgroup = VHGroup('4. Set location to save analysis', 'G')
        analysis_vgroup.gbox.setMaximumHeight(100)
        self.tabs.add_named_tab('Data', analysis_vgroup.gbox)

        self.btn_select_analysis = QPushButton("Set analysis folder")
        self.display_analysis_folder = QLabel("No selection")
        self.display_analysis_folder.setWordWrap(True)
        #, self.scroll_analysis = scroll_label('No selection.')
        analysis_vgroup.glayout.addWidget(self.display_analysis_folder, 0, 0)
        analysis_vgroup.glayout.addWidget(self.btn_select_analysis, 0, 1)

        # load analysis
        self.btn_load_analysis = QPushButton("Load analysis")
        analysis_vgroup.glayout.addWidget(self.btn_load_analysis, 1, 0)

        # segmentation tab
        self.segoptions_vgroup = VHGroup('Set segmentation parameters', orientation='G')
        self.segoptions_vgroup.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Segmentation', self.segoptions_vgroup.gbox)
        # algo choice
        self.seg_algo = QComboBox()
        self.seg_algo.addItems(['conv_paint', 'cellpose', 'precomputed'])
        self.seg_algo.setCurrentIndex(0)
        self.segoptions_vgroup.glayout.addWidget(QLabel('Algorithm'), 0, 0)
        self.segoptions_vgroup.glayout.addWidget(self.seg_algo, 0, 1)

        # cellpose algo options
        self.cellpose_widget = QGroupBox()
        self.cellpose_layout = QGridLayout()
        self.cellpose_widget.setLayout(self.cellpose_layout)
        self.cellpose_widget.setVisible(False)
        self.segoptions_vgroup.glayout.addWidget(self.cellpose_widget, 1, 0, 1, 2)
        self.cell_diameter = QSpinBox()
        self.cell_diameter.setValue(20)
        self.cell_diameter.setMaximum(10000)
        self.cell_diameter_label = QLabel('Cell diameter')
        self.cellpose_layout.addWidget(self.cell_diameter_label, 0, 0, 1, 1)
        self.cellpose_layout.addWidget(self.cell_diameter, 0, 1, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(self.cell_diameter_label, 1, 0, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(self.cell_diameter, 1, 1, 1, 1)
        self.cellpose_flow_threshold = QDoubleSpinBox()
        self.cellpose_flow_threshold.setValue(0.4)
        self.cellpose_flow_threshold.setMaximum(3)
        self.cellpose_flow_threshold.setMinimum(0)
        self.cellpose_flow_threshold.setSingleStep(0.1)
        self.cellpose_layout.addWidget(QLabel('Flow threshold'), 1, 0, 1, 1)
        self.cellpose_layout.addWidget(self.cellpose_flow_threshold, 1, 1, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(QLabel('Flow threshold'), 2, 0, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(self.cellpose_flow_threshold, 2, 1, 1, 1)
        self.cellpose_cellprob_threshold = QDoubleSpinBox()
        self.cellpose_cellprob_threshold.setValue(0.0)
        self.cellpose_cellprob_threshold.setMaximum(6)
        self.cellpose_cellprob_threshold.setMinimum(-6)
        self.cellpose_cellprob_threshold.setSingleStep(0.1)
        self.cellpose_layout.addWidget(QLabel('Cell probability threshold'), 2, 0, 1, 1)
        self.cellpose_layout.addWidget(self.cellpose_cellprob_threshold, 2, 1, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(QLabel('Cell probability threshold'), 3, 0, 1, 1)
        #self.segoptions_vgroup.glayout.addWidget(self.cellpose_cellprob_threshold, 3, 1, 1, 1)

        # convpaint options
        self.conv_paint_widget = ConvPaintWidget(self.viewer)
        self.conv_paint_widget.tabs.setTabVisible(1, False)
        self.conv_paint_widget.update_model_on_project_btn.hide()
        self.conv_paint_widget.prediction_all_btn.hide()
        self.conv_paint_widget.check_use_project.hide()
        self.conv_paint_widget.check_dims_is_channels.hide()
        

        self.segoptions_vgroup.glayout.addWidget(self.conv_paint_widget, 2, 0, 1, 2)
        self.conv_paint_widget.setVisible(True)

        self.segmentation_group = VHGroup('Use saved segmentation', 'G')
        #segmentation_group.glayout.setAlignment(Qt.AlignTop)
        self.segmentation_group.gbox.setVisible(False)
        self.segmentation_group.gbox.setMaximumHeight(150)
        
        self.segoptions_vgroup.glayout.addWidget(self.segmentation_group.gbox, 3, 0, 1, 2)
        self.btn_select_segmentation = QPushButton("Set segmentation folder")
        self.display_segmentation_folder, self.scroll_segmentation = scroll_label('No selection.')
        self.segmentation_group.glayout.addWidget(self.scroll_segmentation, 0, 0)
        self.segmentation_group.glayout.addWidget(self.btn_select_segmentation, 1, 0)

        self.segmentationlayer_group = VHGroup('Use layer', 'G')
        #segmentation_group.glayout.setAlignment(Qt.AlignTop)
        self.segmentationlayer_group.gbox.setVisible(False)
        self.segmentationlayer_group.gbox.setMaximumHeight(150)
        self.segoptions_vgroup.glayout.addWidget(self.segmentationlayer_group.gbox, 4, 0, 1, 2)

        self.pick_layer = create_widget(annotation=napari.layers.Labels, label='Pick segmentation layer')
        self.pick_layer.reset_choices()
        self.viewer.layers.events.inserted.connect(self.pick_layer.reset_choices)
        self.viewer.layers.events.removed.connect(self.pick_layer.reset_choices)
        self.segmentationlayer_group.glayout.addWidget(self.pick_layer.native, 0, 0)
        self.btn_use_layer_as_segmentation = QPushButton("Use layer as segmentation")
        self.segmentationlayer_group.glayout.addWidget(self.btn_use_layer_as_segmentation, 1, 0)

        self.check_use_location = QCheckBox('Select cell')
        self.check_use_location.setChecked(False)
        self.check_use_location.stateChanged.connect(self._on_update_cell_location)
        self.tabs.add_named_tab('Segmentation', self.check_use_location)

        self.btn_run_segmentation = QPushButton("Run segmentation")
        self.tabs.add_named_tab('Segmentation', self.btn_run_segmentation)

        # run analysis
        self.settings_vgroup = VHGroup('Set analysis settings and run', orientation='G')
        self.settings_vgroup.glayout.setAlignment(Qt.AlignTop)
        self.tabs.add_named_tab('Windowing', self.settings_vgroup.gbox)

        # smoothing
        self.smoothing = QSpinBox()
        self.smoothing.setMaximum(1000)
        self.smoothing.setValue(1)
        self.settings_vgroup.glayout.addWidget(QLabel('Smoothing'), 2, 0)
        self.settings_vgroup.glayout.addWidget(self.smoothing, 2, 1)

        ## window options
        self.depth = QSpinBox()
        self.depth.setValue(10)
        self.depth.setMaximum(10000)
        self.settings_vgroup.glayout.addWidget(QLabel('Window depth'), 3, 0)
        self.settings_vgroup.glayout.addWidget(self.depth, 3, 1)
        self.width = QSpinBox()
        self.width.setValue(10)
        self.width.setMaximum(10000)
        self.settings_vgroup.glayout.addWidget(QLabel('Window width'), 4, 0)
        self.settings_vgroup.glayout.addWidget(self.width, 4, 1)

        self.btn_run_spline_and_window = QPushButton("Run windowing")
        self.settings_vgroup.glayout.addWidget(self.btn_run_spline_and_window, 5, 0)
        self.btn_run_single_segmentation = QPushButton("Run single frame")
        self.settings_vgroup.glayout.addWidget(self.btn_run_single_segmentation, 5, 1)
        self.check_use_dask = QCheckBox('Use dask')
        self.check_use_dask.setChecked(False)
        self.settings_vgroup.glayout.addWidget(self.check_use_dask, 5, 2)

        # display options
        self.display_wlayers = QListWidget()
        self.display_wlayers.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.display_wlayers.itemSelectionChanged.connect(self._on_display_wlayers_selection_changed)
        self.tabs.add_named_tab('Signal', QLabel('Window layers'), (0, 0, 1, 1))
        self.tabs.add_named_tab('Signal', self.display_wlayers, (0, 1, 1, 1))
        self.combo_channel = QComboBox()
        self.tabs.add_named_tab('Signal', QLabel('Channel'), (1, 0, 1, 1))
        self.tabs.add_named_tab('Signal', self.combo_channel, (1, 1, 1, 1))
        self.intensity_plot = DataPlotter(self.viewer)
        self.window_loc_plot = None
        self.tabs.add_named_tab('Signal', self.intensity_plot, (2, 0, 1, 2))

        # dask options
        dask_group = VHGroup('Dask options', 'G')
        dask_group.gbox.setMaximumHeight(150)
        self.tabs.add_named_tab('Dask', dask_group.gbox)
        self.dask_num_workers = QSpinBox()
        self.dask_num_workers.setValue(1)
        self.dask_num_workers.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Number of workers'), 0, 0)
        dask_group.glayout.addWidget(self.dask_num_workers, 0, 1)

        self.dask_cores = QSpinBox()
        self.dask_cores.setValue(1)
        self.dask_cores.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Number of cores (SLURM)'), 1, 0)
        dask_group.glayout.addWidget(self.dask_cores, 1, 1)

        self.dask_memory = QSpinBox()
        self.dask_memory.setValue(1)
        self.dask_memory.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Memory per core (SLURM)'), 2, 0)
        dask_group.glayout.addWidget(self.dask_memory, 2, 1)

        self.dask_cluster_type = QComboBox()
        self.dask_cluster_type.addItems(['Local', 'SLURM'])
        self.dask_cluster_type.setCurrentIndex(0)
        dask_group.glayout.addWidget(QLabel('Cluster type'), 3, 0)
        dask_group.glayout.addWidget(self.dask_cluster_type, 3, 1)

        self.dask_initialize_button = QPushButton("Initialize dask")
        self.tabs.add_named_tab('Dask', self.dask_initialize_button)
        
        self.dask_stop_cluster_button = QPushButton("Stop dask cluster")
        self.tabs.add_named_tab('Dask', self.dask_stop_cluster_button)

        # make sure widgets don't occupy more space than they need
        #self._options_layout.addStretch()
        #self._paint_layout.addStretch()
        #self._dask_layout.addStretch()

        # Plot options
        self.drop_choose_plot = QComboBox()
        self.drop_choose_plot.addItems([
            'displacement', 'cumulative displacement', 'curvature',
            'area', 'edge overview'])
        self.drop_choose_plot.setCurrentIndex(0)
        self.tabs.add_named_tab('Morpho', self.drop_choose_plot)
        self.displacement_plot = DataPlotter(self.viewer)
        self.tabs.add_named_tab('Morpho', self.displacement_plot)

        # Correlation options
        self.correlation_plot = DataPlotter(self.viewer)
        self.tabs.add_named_tab('Correlations', self.correlation_plot, (0, 0, 1, 2))
        self.combo_channel_correlation1 = QComboBox()
        self.tabs.add_named_tab('Correlations', QLabel('Channel1'), (1, 0, 1, 1))
        self.tabs.add_named_tab('Correlations', self.combo_channel_correlation1, (1, 1, 1, 1))
        self.combo_channel_correlation2 = QComboBox()
        self.tabs.add_named_tab('Correlations', QLabel('Channel2'), (2, 0, 1, 1))
        self.tabs.add_named_tab('Correlations', self.combo_channel_correlation2, (2, 1, 1, 1))

    
        

        self._add_callbacks()


    def _add_callbacks(self):

        self.seg_algo.currentIndexChanged.connect(self._on_update_param)
        self.depth.valueChanged.connect(self._on_update_param)
        self.width.valueChanged.connect(self._on_update_param)
        self.smoothing.valueChanged.connect(self._on_update_param)

        self.segm_channel.currentItemChanged.connect(self._on_update_param)
        self.signal_channel.itemSelectionChanged.connect(self._on_update_param)

        self.btn_select_file_folder.clicked.connect(self._on_click_select_file_folder)
        self.btn_load_data.clicked.connect(self._on_load_dataset)

        self.btn_select_analysis.clicked.connect(self._on_click_select_analysis)
        self.btn_select_segmentation.clicked.connect(self._on_click_select_segmentation)

        self.btn_load_analysis.clicked.connect(self._on_load_analysis)
        self.combo_channel.currentIndexChanged.connect(self.update_intensity_plot)
        self.combo_channel_correlation1.currentIndexChanged.connect(self.update_correlation_plot)
        self.combo_channel_correlation2.currentIndexChanged.connect(self.update_correlation_plot)

        self.file_list.model().rowsInserted.connect(self._on_change_filelist)
        self.cell_diameter.valueChanged.connect(self._on_update_param)

        self.dask_num_workers.valueChanged.connect(self._on_update_dask_wokers)
        self.dask_initialize_button.clicked.connect(self.initialize_dask)
        self.dask_stop_cluster_button.clicked.connect(self._on_dask_shutdown)

        self.conv_paint_widget.load_model_btn.clicked.connect(self._on_load_model)
        self.conv_paint_widget.save_model_btn.clicked.connect(self._on_load_model)
        self.btn_run_segmentation.clicked.connect(self._on_run_segmentation)
        self.btn_use_layer_as_segmentation.clicked.connect(self._on_use_layer_as_segmentation)
        self.btn_run_single_segmentation.clicked.connect(self._on_run_seg_spline)
        self.btn_run_spline_and_window.clicked.connect(self._on_run_spline_and_window)

        self.drop_choose_plot.currentIndexChanged.connect(self.update_displacement_plot)


    def _on_update_param(self):
        """Update multiple entries of the param object."""
        
        self.param.seg_algo = self.seg_algo.currentText()
        if self.param.seg_algo != 'cellpose':
            #self.cell_diameter.setVisible(False)
            #self.cell_diameter_label.setVisible(False)
            self.cellpose_widget.setVisible(False)
        else:
            #self.cell_diameter.setVisible(True)
            #self.cell_diameter_label.setVisible(True)
            self.cellpose_widget.setVisible(True)
        if self.param.seg_algo != 'conv_paint':
            self.conv_paint_widget.setVisible(False)
        else:
            self.conv_paint_widget.setVisible(True)
        if self.param.seg_algo != 'precomputed':
            self.segmentation_group.gbox.setVisible(False)
            self.segmentationlayer_group.gbox.setVisible(False)
        else:
            self.segmentation_group.gbox.setVisible(True)
            self.segmentationlayer_group.gbox.setVisible(True)

        self.param.lambda_ = self.smoothing.value()
        self.param.width = self.width.value()
        self.param.depth = self.depth.value()
        if self.segm_channel.currentItem() is not None:
            self.param.morpho_name = self.segm_channel.currentItem().text()
            self.conv_paint_widget.param.channel = self.param.morpho_name
        if len(self.signal_channel.selectedItems()) != 0:
            self.param.signal_name = [x.text() for x in self.signal_channel.selectedItems()]
        else:
            self.param.signal_name = []
        
        if self.param.data_type == 'zarr':
            if self.file_list.currentItem() is not None:
                self.param.data_folder = Path(self.file_list.folder_path).joinpath(self.file_list.currentItem().text())
        else:
            if self.file_list.folder_path is not None:
                self.param.data_folder = Path(self.file_list.folder_path)
        
        if self.display_analysis_folder.text() != 'No selection.':
            self.param.analysis_folder = Path(self.display_analysis_folder.text())
        if self.display_segmentation_folder.text() != 'No selection.':
            self.param.seg_folder = Path(self.display_segmentation_folder.text())
        self.param.diameter = self.cell_diameter.value()

    def _on_update_cell_location(self, event=None):

        if self.check_use_location.isChecked():
            if 'select cell' not in self.viewer.layers:
                self.viewer.add_points(
                    np.array([[0]+list(self.data.dims)]) // 2, size=10, face_color='red', name='select cell')
                self.viewer.layers['select cell'].events.data.connect(self._on_update_cell_location)
            self.param.location = list(self.viewer.layers['select cell'].data[0][-2::])
        else:
            if 'select cell' in self.viewer.layers:
                self.viewer.layers.remove('select cell')
            self.param.location = None

    def _on_click_select_file_folder(self, event=None, file_folder=None):
        """Interactively select folder to analyze"""
        
        if file_folder is None:
            if self.qcombobox_data_type.currentText() in ['zarr', 'multipage_tiff', 'tiff_series']:
                file_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            elif self.qcombobox_data_type.currentText() == 'nparray':
                file_folder, ok = QFileDialog.getOpenFileName(self, "Select File")
                
        if self.qcombobox_data_type.currentText() == 'zarr':
            parent_folder = Path(file_folder).parent
            self.file_list.update_from_path(parent_folder)
            items = [self.file_list.item(x).text() for x in range(self.file_list.count())]
            file_index = items.index(Path(file_folder).name)
            self.file_list.setCurrentRow(file_index)
            self.param.data_type = "zarr"
            self.param.data_folder = Path(file_folder)
            self.data, self.param = dataset_from_param(self.param)
        
        elif self.qcombobox_data_type.currentText() == 'multipage_tiff':
            self.param.data_folder = file_folder
            self.file_list.update_from_path(file_folder)
            self.param.data_type = "multi"
            self.data, self.param = dataset_from_param(self.param)
        
        elif self.qcombobox_data_type.currentText() == 'tiff_series':
            self.param.data_folder = file_folder
            self.file_list.update_from_path(file_folder)
            self.param.data_type = "series"
            self.data, self.param = dataset_from_param(self.param)
        
        elif self.qcombobox_data_type.currentText() == 'nparray':
            parent_folder = Path(file_folder).parent
            self.param.data_folder = parent_folder
            self.file_list.update_from_path(parent_folder)
            items = [self.file_list.item(x).text() for x in range(self.file_list.count())]
            file_index = items.index(Path(file_folder).name)
            self.file_list.setCurrentRow(file_index)
            self.param.data_type = "np"
            self.data, self.param = dataset_from_param(self.param)

        elif self.qcombobox_data_type.currentText() == 'layers':
            #self.param.data_folder = file_folder
            #self.file_list.add_elements([x.name for x in self.viewer.layers])
            #self.param.data_type = "zarr"
            #self.data, self.param = dataset_from_param(self.param)
            pass

        # create channel choice
        if self.qcombobox_data_type.currentText() in ['zarr', 'multipage_tiff', 'tiff_series', 'nparray']:
            self._on_load_single_file_data()
        elif self.qcombobox_data_type.currentText() == 'layers':
            self._on_load_layers()


    def _on_change_filelist(self):
        """Update the channel list when main file list changes."""
        
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        self.segm_channel.clear()
        self.signal_channel.clear()
        self.segm_channel.addItems(files)
        self.signal_channel.addItems(files)
        self._on_update_param()

    def _on_load_single_file_data(self):

        channel_list = self.data.channel_name
        self.segm_channel.clear()
        self.signal_channel.clear()
        self.segm_channel.addItems(channel_list)
        self.signal_channel.addItems(channel_list)
        self._on_update_param()

    def _on_load_layers(self):
            
        channel_list = [x.name for x in self.viewer.layers]
        self.segm_channel.clear()
        self.signal_channel.clear()
        self.segm_channel.addItems(channel_list)
        self.signal_channel.addItems(channel_list)
        self._on_update_param()

    def _on_load_model(self):
        self.param.random_forest = Path(self.conv_paint_widget.param.random_forest)

    def _on_run_spline_and_window(self):
        """Run full morphodynamics analysis"""

        self.display_wlayers.clear()

        if self.cluster is None and self.check_use_dask.isChecked():
            self.initialize_dask()
        
        # run with dask if selected
        if self.check_use_dask.isChecked():
            with Client(self.cluster) as client:
                spline_and_window(data=self.data,
                    param=self.param, res=self.res, client=client)
        else:
            spline_and_window(data=self.data,
                    param=self.param, res=self.res)
        
        self._on_load_windows()
        export_results_parameters(self.param, self.res)
        self.update_displacement_plot()

    def _on_run_full_analysis(self):
        """Run full morphodynamics analysis. Currently unused."""

        if self.cluster is None and self.check_use_dask.isChecked():
            self.initialize_dask()
        
        model = None

        # run with dask if selected
        if self.check_use_dask.isChecked():
            with Client(self.cluster) as client:
                self.res = analyze_morphodynamics(
                    data=self.data,
                    param=self.param,
                    client=client,
                    model=model
                )
        else:
            self.res = analyze_morphodynamics(
                    data=self.data,
                    param=self.param,
                    client=None,
                    model=model
                )
        
        self._on_load_windows()
        export_results_parameters(self.param, self.res)
        self.update_displacement_plot()

    def _on_use_layer_as_segmentation(self):
        """Use currently selected layer as segmentation."""

        folder_export = self.param.analysis_folder.joinpath('main_segmentation')
        self.param.seg_folder = folder_export
        self.display_segmentation_folder.setText(str(folder_export))
        if self.param.analysis_folder is None:
            raise ValueError('Select an analysis folder first.')
        if not folder_export.exists():
            folder_export.mkdir()
        layer = self.pick_layer.value
        if layer is None:
            raise ValueError('Select a layer first.')
        if layer.data.shape != self.viewer.layers[self.param.morpho_name].data.shape:
            raise ValueError('Segmentation layer must have same shape as data.')
        
        for k in range(layer.data.shape[0]):
            skimage.io.imsave(
                folder_export.joinpath(f"segmented_k_{k}.tif"),
                layer.data[k], check_contrast=False)


    def _on_run_segmentation(self):
        """Run segmentation."""

        if self.analysis_path is None:
            self._on_click_select_analysis()
            
        if self.cluster is None and self.check_use_dask.isChecked():
            self.initialize_dask()
        
        if self.param.seg_algo == 'conv_paint':
            if self.param.random_forest is None:
                if self.conv_paint_widget.param.random_forest is None:
                    self.conv_paint_widget.save_model()
                self.param.random_forest = self.conv_paint_widget.param.random_forest

        if self.check_use_dask.isChecked():
            with Client(self.cluster) as client:
                self.res, _ = segment_and_track(
                    data=self.data,
                    param=self.param,
                    client=client,
                    cellpose_kwargs={
                        'flow_threshold': self.cellpose_flow_threshold.value(),
                        'cellprob_threshold': self.cellpose_cellprob_threshold.value()
                        }
                )
        else:
            self.res, _ = segment_and_track(
                    data=self.data,
                    param=self.param,
                    client=None,
                    cellpose_kwargs={
                        'flow_threshold': self.cellpose_flow_threshold.value(),
                        'cellprob_threshold': self.cellpose_cellprob_threshold.value()
                        }
                )
        
        self.display_mask()
        export_results_parameters(self.param, self.res)
                            
    def display_mask(self):

        save_path = self.param.analysis_folder.joinpath("segmented")
        mask_list = []
        for k in range(self.data.num_timepoints):
            image_path = save_path.joinpath(f"tracked_k_{k}.tif")  
            mask_list.append(skimage.io.imread(image_path))
        mask_list = np.stack(mask_list, axis=0)
        self.viewer.add_labels(mask_list, name='segmentation')


    def _on_run_seg_spline(self):
        
        if self.analysis_path is None:
            self._on_click_select_analysis()
        
        step = self.viewer.dims.current_step[0]
        image, c, im_windows, windows = compute_spline_windows(self.param, step)

        layer_indices= self._get_layer_indices(windows)
        col_dict, _ = self._create_color_shadings(layer_indices)

        self.viewer.add_labels(image, name='segmentation')
        self.viewer.add_labels(im_windows, name='windows')
        self.viewer.layers['windows'].color = col_dict
        self.viewer.layers['windows'].color_mode = 'direct' 
        self.viewer.add_shapes(
            data=[np.c_[c[1], c[0]]], shape_type='polygon', 
            edge_color='red', face_color=[0,0,0,0], edge_width=1,
            name='spline')


    def _on_segment_single_frame(self):
        """Segment single frame."""
        
        step = self.viewer.dims.current_step[0]
        temp_image = segment_single_frame(
            self.param, step, self.param.analysis_folder, return_image=True,
            cellpose_kwargs={
                        'flow_threshold': self.cellpose_flow_threshold.value(),
                        'cellprob_threshold': self.cellpose_cellprob_threshold.value()
                        }
                    )
        
        self.viewer.add_labels(temp_image)
        #self.viewer.open(self.param.analysis_folder.joinpath("segmented_k_" + str(step) + ".tif"))

    def initialize_dask(self, event=None):
        """Initialize dask client.
        To do: add SLURMCluster and and interface for it"""

        if self.dask_cluster_type.currentText() == 'Local':
            self.cluster = LocalCluster()#n_workers=self.dask_num_workers.value())
            self.dask_num_workers.setValue(len(self.cluster.scheduler_info['workers']))
        elif self.dask_cluster_type.currentText() == 'SLURM':
            self.cluster = SLURMCluster(cores=self.dask_cores.value(), memory=self.dask_memory.value())

    def _on_update_dask_wokers(self):
        """Update dask workers."""
        
        if self.dask_cluster_type.currentText() == 'Local':
            if self.cluster is None:
                self.initialize_dask()
                self.cluster = LocalCluster(n_workers=self.dask_num_workers.value())

            self.cluster.scale(self.dask_num_workers.value())

    def _on_dask_shutdown(self):
        """Shutdown dask workers."""
        
        self.cluster.close()
        self.cluster = None

    def _on_click_select_analysis(self, event=None, analysis_path=None):
        """Select folder where to save the analysis."""

        if analysis_path is not None:
            self.analysis_path = Path(analysis_path)
        else:
            self.analysis_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.display_analysis_folder.setText(self.analysis_path.as_posix())
        self._on_update_param()
        if self.param.seg_folder is None:
            self.param.seg_folder = self.analysis_path.joinpath('main_segmentation')
            self.display_segmentation_folder.setText(self.param.seg_folder.as_posix())

    def _on_click_select_segmentation(self):
        """Select folder where to save the segmentation."""

        self.segmentation_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.display_segmentation_folder.setText(self.segmentation_path.as_posix())
        self._on_update_param()

    def load_convpaint_model(self, return_model=True):
        """Load RF model for segmentation"""
        
        model = Classifier(self.param.random_forest)
        
        if return_model:
            return model
        else:
            return None

    def _on_load_dataset(self):
        """Having selected segmentation and signal channels, load the data"""
        
        if self.qcombobox_data_type.currentText() == 'layers':
            self._convert_layers_to_zarr()

        self.data, self.param = dataset_from_param(self.param)
        self.create_stacks()
        self.combo_channel.addItems(self.data.channel_name)
        self.combo_channel_correlation1.addItems(self.param.signal_name+['displacement'])
        self.combo_channel_correlation2.addItems(self.param.signal_name)

    def _convert_layers_to_zarr(self):
        """Convert layers selected for segmentation and signal to a zarr file
        and save it in the analysis folder. At the end re-load data as standard zarr file."""

        segm_layer = self.segm_channel.currentItem().text()
        signal_layers = [x.text() for x in self.signal_channel.selectedItems()]
        layers = [segm_layer] + signal_layers
        # keep unique layers
        layers = list(dict.fromkeys(layers))
        # get position of layers in global layer list to account for a layer both in segmentation and signal
        signal_id = [layers.index(x) for x in signal_layers]

        np_array = [self.viewer.layers[x].data for x in layers]
        np_array = np.array(np_array)
        import zarr
        if self.analysis_path is None:
            self.analysis_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        z1 = zarr.open(self.analysis_path.joinpath('layer_data.zarr'), mode='w', shape=np_array.shape, chunks=(np_array.shape[0],1,np_array.shape[-2], np_array.shape[-1]))
        z1[:] = np_array
        self.viewer.layers.clear()

        self.qcombobox_data_type.setCurrentText('zarr')
        self._on_click_select_file_folder(file_folder=self.analysis_path.joinpath('layer_data.zarr'))

        self.segm_channel.setCurrentRow(0)
        for i in signal_id:
            self.signal_channel.item(i).setSelected(True)
        
        self._on_click_select_analysis(analysis_path=self.analysis_path)

    def _on_display_wlayers_selection_changed(self):
        """Hide/reveal window layers."""
        
        on_list = [int(x.text()) for x in self.display_wlayers.selectedItems()]
        for i in self.layer_indices:
            if i not in on_list:
                val_to_set = 0
            else:
                val_to_set = 1
            for j in self.layer_global_indices[i]:
                self.viewer.layers['windows'].color[j][-1]=val_to_set
        self.viewer.layers['windows'].color_mode = 'direct'

        self.update_intensity_plot()


    def _get_current_layer_display(self):
        """Get current layer display."""

        on_list = [int(x.text()) for x in self.display_wlayers.selectedItems()]
        return on_list

    def update_intensity_plot(self):

        on_list = self._get_current_layer_display()
        if len(on_list) == 0:
            return
        
        channel_index = self.combo_channel.currentIndex()
        self.intensity_plot.axes.clear()
        self.window_loc_plot = None
        self.intensity_plot.axes.set_title('Intensity')
        self.intensity_plot.axes.set_xlabel('Time')
        self.intensity_plot.axes.set_ylabel('Window')

        signal = self.res.mean[channel_index, on_list[0]]
        percentile1 = np.percentile(signal[~np.isnan(signal)],1)
        percentile99 = np.percentile(signal[~np.isnan(signal)],99)
        self.intensity_plot.axes.imshow(
            signal,
            vmin=percentile1,
            vmax=percentile99)
        
        self.intensity_plot.axes.xaxis.label.set_size(12)
        self.intensity_plot.axes.yaxis.label.set_size(12)
        self.intensity_plot.axes.title.set_fontsize(12)
        self.intensity_plot.axes.tick_params(labelsize=12)
        self.intensity_plot.canvas.figure.canvas.draw()

    def update_correlation_plot(self):
            
        on_list = self._get_current_layer_display()
        if len(on_list) == 0:
            return

        channel_value1 = self.combo_channel_correlation1.currentText()
        channel_value2 = self.combo_channel_correlation2.currentText()

        sel_layer = on_list[0]
        
        self.correlation_plot.canvas.figure.clear()
        fig = self.correlation_plot.canvas.figure
        ax = self.correlation_plot.canvas.figure.subplots()
        show_correlation_core(
            res=self.res, param=self.param, signal1_name=channel_value1,
            signal2_name=channel_value2,
            window_layer=sel_layer,
            normalization='Pearson', fig_ax=(fig, ax))
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_fontsize(12)
        ax.tick_params(labelsize=12)

        self.correlation_plot.canvas.figure.canvas.draw()


    def update_displacement_plot(self):

        self.displacement_plot.canvas.figure.clear()
            
        fig = self.displacement_plot.canvas.figure
        ax = self.displacement_plot.canvas.figure.subplots()
        if self.drop_choose_plot.currentText() == 'displacement':
            show_displacement(self.res, fig_ax=(fig, ax))
        elif self.drop_choose_plot.currentText() == 'curvature':
            show_curvature(self.data, self.res, fig_ax=(fig, ax),show_colorbar=False)
        elif self.drop_choose_plot.currentText() == 'edge overview':
            show_edge_overview(param=self.param, data=self.data, res=self.res, fig_ax=(fig, ax), lw=0.8)
        elif self.drop_choose_plot.currentText() == 'cumulative displacement':
            show_cumdisplacement(self.res, fig_ax=(fig, ax))
        elif self.drop_choose_plot.currentText() == 'area':
            show_geometry(self.data, self.res, prop='area', title='Area [px]', fig_ax=(fig, ax))
            ax.set_xlabel('Time [frame]')
            ax.set_ylabel('Area [px]')

        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_fontsize(12)
        ax.tick_params(labelsize=12)
        self.displacement_plot.canvas.figure.canvas.draw()
        

    def create_stacks(self):
        """Create and add to the viewer datasets as dask stacks.
        Note: this should be added to the dataset class."""

        sample = self.data.load_frame_morpho(0)

        if self.data.data_type == 'h5':
            h5file = self.data.channel_imobj[self.data.channel_name.index(self.param.morpho_name)]
            seg_stack = da.from_array(h5file)
        else:
            my_data = self.data
            def return_image(i):
                return my_data.load_frame_morpho(i)
            seg_lazy_arrays = [delayed(return_image)(i) for i in range(self.data.max_time)]
            dask_seg_arrays = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in seg_lazy_arrays]
            seg_stack = da.stack(dask_seg_arrays, axis=0)
        self.viewer.add_image(data=seg_stack,
                              name=self.param.morpho_name,
                              colormap='gray',
                              blending='additive')

        colorlist = ['magenta', 'yellow', 'cyan', 'green', 'red', 'blue']
        for ind, c in enumerate(self.param.signal_name):
            if self.data.data_type == 'h5':
                h5file = self.data.channel_imobj[self.data.channel_name.index(c)]
                sig_stack = da.from_array(h5file)
            else:
                my_data = self.data
                def return_image(ch, t):
                    return my_data.load_frame_signal(ch, t)

                sig_lazy_arrays = [delayed(return_image)(ind, i) for i in range(self.data.max_time)]
                dask_sig_arrays = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in sig_lazy_arrays]
                sig_stack = da.stack(dask_sig_arrays, axis=0)
            self.viewer.add_image(
                data=sig_stack, 
                name=f'signal {c}',
                colormap=colorlist[ind],
                blending='additive')
    
    def create_stacks_classical(self):
        """Create stacks from the data and add them to viewer."""
        
        seg_stack = np.stack(
            [self.data.load_frame_morpho(i) for i in range(self.data.max_time)], axis=0)
        self.viewer.add_image(seg_stack, name=self.param.morpho_name)

        sig_stack = [np.stack(
            [self.data.load_frame_signal(c, i) for i in range(self.data.max_time)], axis=0)
            for c in range(len(self.param.signal_name))]
        for ind, c in enumerate(self.param.signal_name):
            self.viewer.add_image(sig_stack[ind], name=f'signal {c}')

    def _get_layer_indices(self, windows):
        """Given a windows list of lists, create a dictionary where each entry i
        contains the labels of all windows in layer i"""

        layer_indices= {}
        count=1
        for i in range(len(windows)):
            gather_indices=[]
            for j in range(len(windows[i])):
                gather_indices.append(count)
                count+=1
            layer_indices[i]=gather_indices
        return layer_indices

    def _create_color_shadings(self, layer_index):
        """Given a layer index (from _get_layer_indices), create a dictionary where
        each entry i contains the colors for label i. Colors are in shades of a given
        color per layer."""

        ## create a color dictionary, containing for each label index a color
        ## currently each layer gets a color and labeles within it get a shade of that color
        color_layers = ['red', 'blue', 'cyan', 'magenta']
        color_pool = cycle(color_layers)
        global_index=1
        col_dict = {None: np.array([0., 0., 0., 1.], dtype=np.float32)}
        layer_global_indices = {i: [] for i in range(len(layer_index))}
        for (lay, col) in zip(layer_index, color_pool):
            num_colors = len(layer_index[lay])
            #color_array = napari.utils.colormaps.SIMPLE_COLORMAPS[col].map(np.linspace(0.1,1,num_colors))
            color_array = cycle(napari.utils.colormaps.SIMPLE_COLORMAPS[col].map(np.array([0.3,0.45,0.6,0.75, 0.9])))
            for ind2, col_index in zip(range(num_colors), color_array):
                col_dict[global_index]=col_index
                layer_global_indices[lay].append(global_index)
                global_index+=1
        return col_dict, layer_global_indices

    def _on_load_windows(self):
        """Add windows labels to the viewer"""

        # create array to contain the windows
        w_image = np.zeros((
            self.data.max_time,
            self.viewer.layers[self.param.morpho_name].data.shape[1],
            self.viewer.layers[self.param.morpho_name].data.shape[2]), dtype=np.uint16)
        
        # load window indices and use them to fill window array
        # keep track of layer to which indices belong in self.layer_indices
        for t in range(self.data.max_time):
            name = Path(self.param.analysis_folder).joinpath(
                'segmented', "window_k_" + str(t) + ".pkl")
            windows = pickle.load(open(name, 'rb'))
            if t==0:
                self.layer_indices= self._get_layer_indices(windows)

            w_image[t, :, :] = label_windows(
                shape=(self.data.shape[0], self.data.shape[1]), windows=windows)

        col_dict, self.layer_global_indices = self._create_color_shadings(self.layer_indices)
        
        # assign color dictionary to window layer colormap
        self.viewer.add_labels(w_image, name='windows')
        self.viewer.layers['windows'].color = col_dict
        self.viewer.layers['windows'].color_mode = 'direct' #needed to refresh the color map
        self.viewer.layers['windows'].mouse_move_callbacks.append(self._shift_move_callback)

        self.display_wlayers.addItems([str(x) for x in self.layer_indices.keys()])

    def _on_load_analysis(self):
        """Load existing output of analysis"""
        self.display_wlayers.clear()
        self.combo_channel.clear()
        self.combo_channel_correlation1.clear()
        self.combo_channel_correlation2.clear()
        
        if self.analysis_path is None:
            self._on_click_select_analysis()

        self.param, self.res, self.data = load_alldata(
            self.analysis_path, load_results=True
        )
        self.file_list.model().rowsInserted.disconnect(self._on_change_filelist)
        self.segm_channel.currentItemChanged.disconnect(self._on_update_param)
        self.signal_channel.itemSelectionChanged.disconnect(self._on_update_param)
        
        if self.param.data_type == 'zarr':
            main_folder = self.param.data_folder.parent
        else:
            main_folder = self.param.data_folder
        
        self.file_list.update_from_path(main_folder)
        # select an element in the file list
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        self.file_list.item(files.index(self.param.data_folder.name)).setSelected(True)

        self.segm_channel.addItems(self.data.channel_name)
        self.segm_channel.item(self.data.channel_name.index(self.param.morpho_name)).setSelected(True)
        self.signal_channel.addItems(self.data.channel_name)
        self.combo_channel.addItems(self.data.channel_name)
        self.combo_channel_correlation1.addItems(self.param.signal_name+['displacement'])
        self.combo_channel_correlation2.addItems(self.param.signal_name)

        for i in range(len(self.param.signal_name)):
            self.signal_channel.item(self.data.channel_name.index(self.param.signal_name[i])).setSelected(True)
        
        self.file_list.model().rowsInserted.connect(self._on_change_filelist)
        self.segm_channel.currentItemChanged.connect(self._on_update_param)
        self.signal_channel.itemSelectionChanged.connect(self._on_update_param)

        self._on_update_interface()
        self.create_stacks()

        if len(list(Path(self.param.analysis_folder).joinpath('segmented').glob('*.pkl'))) > 0:
            self._on_load_windows()
        else:
            self.display_mask()

        
        #plots
        self.update_displacement_plot()
        self.update_correlation_plot()
        

    def _on_update_interface(self):
        """Update UI when importing existing analyis"""

        self.seg_algo.setCurrentText(self.param.seg_algo)
        self.cell_diameter.setValue(self.param.diameter)

    def _shift_move_callback(self, layer, event):
        """Receiver for napari.viewer.mouse_move_callbacks, checks for 'Shift' event modifier.
        If event contains 'Shift' and layer attribute contains napari layers the cursor position
        and value is used to indicate the current position in the window intensity plot.
        """

        if 'Shift' in event.modifiers:
            data_coordinates = layer.world_to_data(event.position)
            val = layer.get_value(data_coordinates)

            on_list = self._get_current_layer_display()
            val = val -1 - on_list[0] * self.res.mean.shape[2]
            
            if self.window_loc_plot is not None:
                self.window_loc_plot[0].set_data(([data_coordinates[0]], [val]))
    
            else:
                self.window_loc_plot = self.intensity_plot.axes.plot([data_coordinates[0]], [val], 'ro')
            
            self.intensity_plot.canvas.draw()

def scroll_label(default_text = 'default text'):
    mylabel = QLabel()
    mylabel.setText('No selection.')
    myscroll = QScrollArea()
    myscroll.setWidgetResizable(True)
    myscroll.setWidget(mylabel)
    return mylabel, myscroll