import ipywidgets as widgets
import numpy as np
layout = widgets.Layout(width='auto', height='40px')
def get_options(file_choice):
    all_parameter_widgets = [widgets.Label(value='Pick Parameters')]

    angles_dict = {"Every Angle Increment": None, 
                   "Every 2nd Angle Increment": slice(0,-1,2),
                   "Every 10th Angle Increment": slice(0,-1,10),
                   "Every 20th Angle Increment": slice(0,-1,20)
    }
    angle_options = widgets.Select(options=angles_dict.keys(), layout={'width': 'max-content'})

    def select_angles(angles):
        ang = angles_dict[angles]
        print("Selected:", angles)
        return ang

    angles_choice = widgets.interactive(select_angles, angles = angle_options, description = "Pick Angles")
    all_parameter_widgets.append(angles_choice)
    ########################################################################################################


    slices_dict = {
        "Middle Slice": slice(int(np.ceil(file_choice.result[1]["numslices"]/2)),int(np.ceil(file_choice.result[1]["numslices"]/2))+1,1),
        "10 Middle Slices": slice(int(np.ceil(file_choice.result[1]["numslices"]/2))-5,int(np.ceil(file_choice.result[1]["numslices"]/2))+5,1),
        "20 Middle Slices": slice(int(np.ceil(file_choice.result[1]["numslices"]/2))-10,int(np.ceil(file_choice.result[1]["numslices"]/2))+10,1)
    }
    slice_options = widgets.Select(options=slices_dict.keys(), value="Middle Slice", layout={'width': 'max-content'}, description = "Pick Slices")

    def select_slices(slices):
        s = slices_dict[slices]
        print("Selected:", slices)
        return s

    slices_choice = widgets.interactive(select_slices, slices = slice_options, display='flex',
        flex_flow='column',
        align_items='stretch', 
        layout = layout, style={'description_width': 'initial'})
    all_parameter_widgets.append(slices_choice)
    ########################################################################################################


    downsamp = widgets.BoundedIntText(
        value=2,
        min=1,
        max=100,
        description='Downsample Factor:', style={'description_width': 'initial'}
    )

    def enter_downsample_factor(factor):
        print("Downsampling by: ", factor)
        return factor

    downsamp_choice = widgets.interactive(enter_downsample_factor, factor = downsamp, flex_flow='column',
        align_items='stretch', 
        layout = layout, style={'description_width': 'initial'})
    all_parameter_widgets.append(downsamp_choice)
    ########################################################################################################

    cor_range = widgets.BoundedIntText(
        value=10,
        min=1,
        max=100,
        description='Center of Rotation Search Range:', style={'description_width': 'initial'}
    )

    def enter_corsearch_factor(factor):
        print("Center of Rotation Search Range chosen: ", factor)
        return factor

    corsearch_choice = widgets.interactive(enter_corsearch_factor, factor = cor_range, flex_flow='column',
        align_items='stretch', 
        layout = layout, style={'description_width': 'initial'})

    all_parameter_widgets.append(corsearch_choice)
    ########################################################################################################

    cor_step = widgets.BoundedFloatText(
        value=0.5,
        min=0.25,
        max=5,
        step = 0.05,
        description='Center of Rotation Search Step size:', style={'description_width': 'initial'}
    )

    def enter_corstep_factor(factor):
        print("Center of Rotation Search Range chosen: ", factor)
        return factor

    corstep_choice = widgets.interactive(enter_corstep_factor, factor = cor_step, flex_flow='column',
        align_items='stretch', 
        layout = layout, style={'description_width': 'initial'})


    all_parameter_widgets.append(corstep_choice)
    ########################################################################################################

    fc = widgets.BoundedFloatText(
        value=1,
        min=0,
        max=1,
        step = 0.001,
        description='Filter Cutoff (0 - 1, 1 is no filtering):', style={'description_width': 'initial'}
    )

    def enter_fc_factor(factor):
        print("Filter cutoff chosen: ", factor)
        return factor

    fc_choice = widgets.interactive(enter_fc_factor, factor = fc, flex_flow='column',
        align_items='stretch', 
        layout = layout, style={'description_width': 'initial'})


    all_parameter_widgets.append(fc_choice)
    ########################################################################################################
    return all_parameter_widgets