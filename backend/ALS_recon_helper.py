import numpy as np
import ipywidgets as widgets
import ALS_recon_functions as als


def default_reconstruction(path, angles_ind, slices_ind, proj_downsample, COR, fc,
                           preprocessing_args=None, postprocessing_args=None, use_gpu=False):
    """
    This is what ALS_recon notebook calls -- can use fbp, cgls, or something else
    """
    tomo, angles = als.read_data(path,
                                 proj=angles_ind, sino=slices_ind,
                                 downsample_factor=proj_downsample,
                                 preprocess_settings=preprocessing_args,
                                 postprocess_settings=postprocessing_args)
    recon = als.astra_fbp_recon(tomo, angles, COR=COR/proj_downsample, fc=fc, gpu=use_gpu)
    # recon = als.astra_cgls_recon(tomo, angles, COR=COR/proj_downsample, num_iter=20, gpu=use_gpu)
    return recon, tomo

def show_slice_reconstruction(path, slice_num,
                              proj_downsample, angles_downsample,
                              COR, fc,
                              minimum_transmission,
                              outlier_diff, outlier_size,
                              sarepy_snr, sarepy_la_size, sarepy_sm_size,
                              ringSigma, ringLevel,
                              use_gpu,
                              img_handle,
                              sino_handle,
                              hline_handle):
    
    slices_ind = slice(slice_num,slice_num+1,1)
    angles_ind = slice(0,-1,angles_downsample)
    preprocessing_args = {"minimum_transmission": minimum_transmission,
                          "snr": sarepy_snr,
                          "la_size": sarepy_la_size,
                          "sm_size": sarepy_sm_size,
                          "outlier_diff_1D": outlier_diff,
                          "outlier_size_1D": outlier_size
                         }
    postrocessing_args = {"ringSigma": ringSigma,
                          "ringLevel": ringLevel
                         }
    recon, tomo = default_reconstruction(path, angles_ind, slices_ind, proj_downsample, COR, fc, preprocessing_args, postrocessing_args, use_gpu)
    img_handle.set_data(recon.squeeze())
    sino_handle.set_data(tomo.squeeze())
    hline_handle.set_ydata([slice_num,slice_num])

def reconstruction_parameter_options(path,metadata,cor_init,use_gpu,img_handle,sino_handle,hline_handle):
    """
        Create widgets for every parameter, then put into Tabs widgets and interactive_output
    """
    ################## Common Parameters ##################################    
    parameter_widgets = {}
    # Angle downsample    
    angle_downsample_widget = widgets.Dropdown(
        options=[("Every Angle",1), ("Every 2nd Angle",2), ("Every 4th Angle",4), ("Every 8th Angle",8)],
        value=1,
        description='Angle Downsampling:',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    parameter_widgets['angle_downsample'] = angle_downsample_widget
    # Projection downsample
    proj_downsample_widget = widgets.Dropdown(
        options=[("Full res",1), ("2x downsample",2), ("4x downsample",4), ("8x downsample",8)],
        value=1,
        description='Projection Downsampling:',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    parameter_widgets['proj_downsample'] = proj_downsample_widget
    # COR    
    cor_widget = widgets.FloatSlider(description='COR', layout=widgets.Layout(width='100%'),
                                    min=cor_init - 10,
                                    max=cor_init + 10,
                                    step=0.25,
                                    value=cor_init)
    parameter_widgets['cor'] = cor_widget
    # LP Filter cutoff
    fc_widget = widgets.BoundedFloatText(
        min=0.01,
        max=1,
        step = 0.01,
        value=1,
        description='Filter Cutoff (0.01 - 1, 1 is no filtering):',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    parameter_widgets['fc'] = fc_widget
    # Slice number   
    slice_num_widget = widgets.IntSlider(description='Slice:', layout=widgets.Layout(width='100%'),
                                         min=0,
                                         max=metadata['numslices']-1,
                                         value=metadata['numslices']//2)
    
    parameter_widgets['slice_num'] = slice_num_widget

    ################## Ring Removal Parameters ##################################    
    ringRemoval_parameter_widgets = {}
    # Sarepy small size
    sarepy_small_size_widget = widgets.BoundedIntText(description='Sarepy Small Ring Size:', layout=widgets.Layout(width='90%'),
                                    min=1,
                                    max=41,
                                    step=2,
                                    value=11,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )                                               

    ringRemoval_parameter_widgets['sarepy_small'] = sarepy_small_size_widget
    # Large size (ring removal)    
    sarepy_large_size_widget = widgets.BoundedIntText(description='Sarepy Large Ring Size:', layout=widgets.Layout(width='90%'),
                                    min=1,
                                    max=101,
                                    step=2,
                                    value=1,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    ringRemoval_parameter_widgets['sarepy_large'] = sarepy_large_size_widget
    # SNR (ring removal)    
    sarepy_snr_widget = widgets.BoundedIntText(description='Sarepy SNR:', layout=widgets.Layout(width='90%'),
                                    min=1.1,
                                    max=3,
                                    step=0.1,
                                    value=3,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    ringRemoval_parameter_widgets['sarepy_snr'] = sarepy_snr_widget
    # Wavelet Sigma
    ringSigma_widget = widgets.BoundedFloatText(description='Wavelet Sigma (default: 3)', layout=widgets.Layout(width='90%'),
                                    min=0,
                                    max=10,
                                    step=0.1,
                                    value=0,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    ringRemoval_parameter_widgets['ringSigma'] = ringSigma_widget
    # Wavelet Level
    ringLevel_widget = widgets.BoundedIntText(description='Wavelet Level (default: 8):', layout=widgets.Layout(width='90%'),
                                    min=0.,
                                    max=10.,
                                    step=0.1,
                                    value=0,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    ringRemoval_parameter_widgets['ringLevel'] = ringLevel_widget

    ################## Additional Parameters ##################################    
    additional_parameter_widgets = {}
    # Minimum transmission    
    minTranmission_widget = widgets.BoundedFloatText(description='Min Trans:', layout=widgets.Layout(width='90%'),
                                    min=0.,
                                    max=1.,
                                    step=0.01,
                                    value=0.01,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    additional_parameter_widgets['min_transmission'] = minTranmission_widget
   # Outlier diff along angle
    outlierDiff_widget = widgets.BoundedFloatText(description='Outlier Diff (default 750):', layout=widgets.Layout(width='90%'),
                                    min=0.,
                                    max=10.,
                                    step=0.1,
                                    value=0,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    additional_parameter_widgets['outlier_diff'] = outlierDiff_widget
   # Outlier size along angle
    outlierSize_widget = widgets.BoundedIntText(description='Outlier Size  (default 3):', layout=widgets.Layout(width='90%'),
                                    min=1,
                                    max=21,
                                    step=1,
                                    value=1,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    additional_parameter_widgets['outlier_size'] = outlierSize_widget

    ########################################################################################################
    out = widgets.interactive_output(show_slice_reconstruction,
                            {'path': widgets.fixed(path),
                             'slice_num': slice_num_widget,
                             'angles_downsample': angle_downsample_widget,
                             'proj_downsample': proj_downsample_widget,
                             'COR': cor_widget,
                             'fc': fc_widget,
                             'minimum_transmission': minTranmission_widget,
                             'outlier_diff': outlierDiff_widget,
                             'outlier_size': outlierSize_widget,
                             'sarepy_sm_size': sarepy_small_size_widget,
                             'sarepy_la_size': sarepy_large_size_widget,
                             'sarepy_snr': sarepy_snr_widget,
                             'ringSigma': ringSigma_widget,
                             'ringLevel': ringLevel_widget,
                             'use_gpu': widgets.fixed(use_gpu),
                             'img_handle': widgets.fixed(img_handle),
                             'sino_handle': widgets.fixed(sino_handle),
                             'hline_handle': widgets.fixed(hline_handle)
                            })    
    
    common_box = widgets.VBox(list(parameter_widgets.values()))
    ring_box  = widgets.VBox(list(ringRemoval_parameter_widgets.values()))
    additional_box = widgets.VBox(list(additional_parameter_widgets.values()))
    all_parameters_tab = widgets.Tab(children=[common_box,ring_box,additional_box])
    all_parameters_tab.set_title(0,'Basic Parameters')
    all_parameters_tab.set_title(1,'Ring Removal')
    all_parameters_tab.set_title(2,'Additional Parameters')
    all_parameters_tab.selected_index = 0

    parameter_widgets['ring'] = ringRemoval_parameter_widgets
    parameter_widgets['additional'] = additional_parameter_widgets
    return parameter_widgets, all_parameters_tab, out