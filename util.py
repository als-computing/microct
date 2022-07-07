import numpy as np
import ipywidgets as widgets
import ALS_recon_helper as als


def show_slice_reconstruction(path, slice_num,
                              proj_downsample, angles_downsample,
                              COR, fc,
                              minimum_transmission, snr, la_size, sm_size,
                              use_gpu,
                              img_handle,
                              hline_handle
                            ):
    
    slices_ind = slice(slice_num,slice_num+1,1)
    angles_ind = slice(0,-1,angles_downsample)
    preprocessing_args = {"minimum_transmission": minimum_transmission, "snr": snr, "la_size": la_size, "sm_size": sm_size}
    recon = als.default_reconstruction(path, angles_ind, slices_ind, proj_downsample, COR, fc, preprocessing_args, use_gpu)
    img_handle.set_data(recon.squeeze())
    hline_handle.set_ydata([slice_num,slice_num])

def reconstruction_parameter_options(path,metadata,cor_init,use_gpu,img_handle,hline_handle,label):
    """
        Create widgets for every parameter, then put into interactive_widgets box
    """
    ################## Label ##################################    
    all_parameter_widgets = [widgets.Label(value=label,layout=widgets.Layout(justify_content="center"))]
    ################## Angle downsample ##################################    
    angle_downsample_widget = widgets.Dropdown(
        options=[("Every Angle",1), ("Every 2nd Angle",2), ("Every 4th Angle",4), ("Every 8th Angle",8)],
        value=1,
        description='Angle Downsampling:',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    all_parameter_widgets.append(angle_downsample_widget)
    ################## Projection downsample ##################################    
    proj_downsample_widget = widgets.Dropdown(
        options=[("Full res",1), ("2x downsample",2), ("4x downsample",4), ("8x downsample",8)],
        value=1,
        description='Projection Downsampling:',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    all_parameter_widgets.append(proj_downsample_widget)
    ################## filter cutoff ##################################    
    fc_widget = widgets.BoundedFloatText(
        value=1,
        min=0.01,
        max=1,
        step = 0.01,
        description='Filter Cutoff (0.01 - 1, 1 is no filtering):',
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )

    all_parameter_widgets.append(fc_widget)
    ################## COR ##################################    
    cor_widget = widgets.FloatSlider(description='COR', layout=widgets.Layout(width='100%'),
                                    min=cor_init - 10,
                                    max=cor_init + 10,
                                    step=0.25,
                                    value=cor_init)

    all_parameter_widgets.append(cor_widget)
    ################## Minimum transmission ##################################    
    minTranmission_widget = widgets.BoundedFloatText(description='Min Trans:', layout=widgets.Layout(width='70%'),
                                    min=0.,
                                    max=1.,
                                    step=0.01,
                                    value=0,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    all_parameter_widgets.append(minTranmission_widget)
    ################## Small size (ring removal) ##################################    
    small_size_widget = widgets.BoundedIntText(description='Small Ring Size:', layout=widgets.Layout(width='50%'),
                                    min=1,
                                    max=31,
                                    step=2,
                                    value=11,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )                                               

    all_parameter_widgets.append(small_size_widget)
    ################## Large size (ring removal) ##################################    
    large_size_widget = widgets.BoundedIntText(description='Large Ring Size:', layout=widgets.Layout(width='50%'),
                                    min=1,
                                    max=101,
                                    step=2,
                                    value=1,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    all_parameter_widgets.append(large_size_widget)
    ################## SNR (ring removal) ##################################    
    snr_widget = widgets.BoundedIntText(description='SNR:', layout=widgets.Layout(width='50%'),
                                    min=1.1,
                                    max=3,
                                    step=0.1,
                                    value=3,
        style={'description_width': 'initial'} # this makes sure description text doesn't get cut off
    )
    all_parameter_widgets.append(snr_widget)
    ################## Slice number ##################################    
    slice_num_widget = widgets.IntSlider(description='Slice:', layout=widgets.Layout(width='100%'),
                                         min=0,
                                         max=metadata['numslices']-1,
                                         value=metadata['numslices']//2)
    
    all_parameter_widgets.append(slice_num_widget) 
    ########################################################################################################
    out = widgets.interactive_output(show_slice_reconstruction,
                            {'path': widgets.fixed(path),
                             'slice_num': slice_num_widget,
                             'angles_downsample': angle_downsample_widget,
                             'proj_downsample': proj_downsample_widget,
                             'COR': cor_widget,
                             'fc': fc_widget,
                             'minimum_transmission': minTranmission_widget,
                             'sm_size': small_size_widget,
                             'la_size': large_size_widget,
                             'snr': snr_widget,
                             'use_gpu': widgets.fixed(use_gpu),
                             'img_handle': widgets.fixed(img_handle),
                             'hline_handle': widgets.fixed(hline_handle)
                            }
                           )
    
    return all_parameter_widgets, out