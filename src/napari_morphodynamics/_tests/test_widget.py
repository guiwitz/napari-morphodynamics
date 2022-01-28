import pytest
import napari_morphodynamics
import napari_morphodynamics.napari_gui
from morphodynamics.data import synth
import numpy as np
import h5py

import napari_morphodynamics

@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """Create a dataset for testing."""

    new_path = tmp_path_factory.mktemp("dataset")
    im, sigs = synth.generate_dataset(height=100, width=100, steps=40, step_reverse=20,
                                  displacement=0.5, radius=10, shifts=[3,7])

    for ind, s in enumerate([im]+sigs):
        h5_name = new_path.joinpath(f'synth_ch{ind+1}.h5')
        with h5py.File(h5_name, "w") as f_out:
            dset = f_out.create_dataset("volume", data=s, chunks=True, compression="gzip", compression_opts=1)

    return new_path

@pytest.fixture
def mywidget(make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(napari_morphodynamics, name='napari-morphodynamics')
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name='napari-morphodynamics', widget_name='MorphoWidget',
    )
    return widget

def test_data_exist(dataset):
    """Test that the data exists."""
    assert dataset.joinpath("synth_ch1.h5").is_file()
    assert dataset.joinpath("synth_ch2.h5").is_file()
    assert dataset.joinpath("synth_ch3.h5").is_file()

def test_project_widget(mywidget):
    
    assert isinstance(mywidget, napari_morphodynamics.napari_gui.MorphoWidget)

def test_set_dataset(mywidget, dataset):
    """Test that the dataset is updated."""

    mywidget.file_list.update_from_path(dataset)
    channels = [mywidget.segm_channel.item(i).text() for i in range(mywidget.segm_channel.count())]
    for i in range(3):
        assert f'synth_ch{i+1}.h5' in channels

    assert mywidget.param.data_folder == dataset

def test_load(mywidget, dataset):
    # set path, set channels to use
    mywidget.file_list.update_from_path(dataset)
    channels = [mywidget.segm_channel.item(i).text() for i in range(mywidget.segm_channel.count())]
    mywidget.segm_channel.setCurrentRow(channels.index('synth_ch1.h5'))
    for i in range(mywidget.signal_channel.count()):
        if mywidget.signal_channel.item(i).text() in ['synth_ch2.h5', 'synth_ch3.h5']:
            mywidget.signal_channel.item(i).setSelected(True)

    # load the data
    mywidget._on_load_dataset()

    # check that the data is loaded
    assert mywidget.data.dims == (100,100)
    assert mywidget.param.morpho_name == 'synth_ch1.h5'

    # check that the data is added to viewer
    assert len(mywidget.viewer.layers) == 3
    assert mywidget.viewer.layers[0].name == 'synth_ch1.h5'
    assert mywidget.viewer.layers[1].name == 'signal synth_ch3.h5'
    assert mywidget.viewer.layers[2].name == 'signal synth_ch2.h5'