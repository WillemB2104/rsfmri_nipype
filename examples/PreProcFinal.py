# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:10:00 2016

@author: eaxfjord
"""

import nibabel as nb
from nipype.interfaces.spm import Normalize12, SliceTiming, Realign, Smooth
from nipype.interfaces.spm import Coregister
from SPMCustom import NewSegment
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.fsl as fsl
from nipype.utils.filemanip import split_filename
import numpy as np
from nipype.interfaces.matlab import MatlabCommand
import os
from nipype import config

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           ('from nipype.utils.filemanip import filename_to_list, '
               'list_to_filename, split_filename'),
           'from scipy.special import legendre'
           ]


#config.set('execution', 'remove_unnecessary_outputs', 'False')
config.set('execution', 'single_thread_matlab', 'True')

# Specification to MATLAB
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

fsl.FSLCommand.set_default_output_type('NIFTI')

# Specify Variables
experiment_dir = '/data/eaxfjord/fmriRSWorkingDir/nipype/'
data_dir = os.path.join(experiment_dir, 'data')  # location of the data
working_dir = 'working_dir_PreProc_Final'
output_dir = 'output_dir_PreProc3_Final'


# Tissue probability map
tpm = '/usr/local/matlabtools/2014b/spm12/tpm/TPM.nii'

# Ventricle mask
VentricleMask = '/data/eaxfjord/fmriRSWorkingDir/nipype/data/masks/ventricles/ventricles.nii'

subject_list = os.listdir(os.path.join(data_dir, 'FunImg')) #all subjects

#subject_list = ['pat007_T1', 'pat007_T2']

number_of_slices = 25
TR = 2.0
template = '/usr/local/matlabtools/2014b/spm12/tpm/TPM.nii'
detrend_poly = 1 # remove linear trend
slice_timing= 0 # do slice timing/or not


# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles
templates={"anat": "data/T1Img/{subject_id}/*.nii",
           "func": "data/FunImg/{subject_id}/*.nii"}
selectfiles = Node(SelectFiles(templates), "selectfiles")
selectfiles.inputs.base_directory = experiment_dir


#node that creates 4D files out of 3D files using fsl merge
merge_to_4d = True
merge = Node(interface=fsl.Merge(), name="merge")
merge.inputs.dimension="t"
merge.inputs.tr= TR
merge.inputs.output_type='NIFTI'

#Realign, motion correction
realign = Node(Realign(register_to_mean=True),name="realign")
realign.inputs.jobtype= 'estimate'

meanfunc = Node(fsl.ImageMaths(op_string='-Tmean',
                                            suffix='_mean'),
                   name='meanfunc')

coregister = Node(Coregister(), name="coregister")
coregister.inputs.jobtype = 'estimate'

# Smooth - to smooth the images with a given kernel
smooth = Node(Smooth(), name = "smooth")
smooth.inputs.fwhm = 8

# NewSegment - first step in normalisation
segment = Node(NewSegment(),name="segment")
segment.inputs.channel_info = (0.001, 60, (True, True))
segment.inputs.write_deformation_fields = [True, True]
tissue1 = ((tpm, 1), 1, (True,False), (True, False))
tissue2 = ((tpm, 2), 1, (True,False), (True, False))
tissue3 = ((tpm, 3), 2, (True,False), (True, False))
tissue4 = ((tpm, 4), 3, (False,False), (False, False))
tissue5 = ((tpm, 5), 4, (False,False), (False, False))
tissue6 = ((tpm, 6), 2, (False,False), (False, False))
segment.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
segment.inputs.affine_regularization = 'mni'
segment.inputs.warping_regularization = [0, 0.001, 0.5, 0.05, 0.2]
segment.inputs.cleanup = 1
segment.inputs.warp_fwhm = 0
segment.inputs.sampling_distance = 3

# Normalise - second step in normalisation
normalize_func = Node(Normalize12(jobtype='write',
                      write_voxel_sizes= [4, 4, 4]),
                    name='normalize_func')

normalize_mask = Node(Normalize12(jobtype='write',
                      write_voxel_sizes= [4, 4, 4]),
                    name='normalize_mask')

normalize_struct = Node(Normalize12(jobtype='write',
                                    write_voxel_sizes= [1,1,1]),
                        name='normalize_struct')


# fsl bet
bet_struct = Node(fsl.BET(),name='bet_struct')
bet_struct.inputs.frac = 0.5
bet_struct.inputs.mask = True
bet_struct.inputs.output_type = 'NIFTI'
bet_struct.inputs.robust= True

bet_func= Node(fsl.BET(),name='bet_func')
bet_func.inputs.frac = 0.6
bet_func.inputs.mask = True
bet_func.inputs.output_type = 'NIFTI'
bet_func.inputs.robust= True

#Function to filter 4d data
def bpfilter(in_files,lp,hp,TR):
    from nibabel import load
    import nitime.fmri.io as io
    out_files = []
    path, name, ext = split_filename(in_files)
    out_file = os.path.join(os.getcwd(), name + '_bp' + ext)

    fmri_data = load(in_files)

    volume_shape = fmri_data.shape[:-1]
    coords = list(np.ndindex(volume_shape))
    coords2 = np.array(coords).T

    T_fourier = io.time_series_from_file(in_files,
                              coords2,
                              TR=TR,
                              filter=dict(lb=lp,
                                          ub=hp,
                     mports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           ('from nipype.utils.filemanip import filename_to_list, '
               'list_to_filename, split_filename'),
           'from scipy.special import legendre'
           ]
                     method='fourier'))

    filtered_data = np.zeros(fmri_data.shape)

    idx=0
    for i in coords:
        filtered_data[i]=T_fourier.data[idx]
        idx= idx+1


    img_out = nb.Nifti1Image(filtered_data,fmri_data.get_affine(),fmri_data.get_header())

    img_out.to_filename(out_file)
    out_files.append(out_file)

    return out_files

#Function node to filter both smooth and unsmooth
bandpass = Node(Function(input_names=['in_files','lp','hp','TR'],
                         output_names= 'out_files',
                         function = bpfilter,imports=imports),name='bandpass')

bandpass.inputs.lp = 0.01
bandpass.inputs.hp = 0.1
bandpass.inputs.TR = TR

bandpass_unsmooth = bandpass.clone('bandpass_unsmooth')

#create the preprocessing workflow
preproc = Workflow(name='preproc3')
preproc.base_dir = os.path.join(experiment_dir,working_dir)

preproc.connect([(infosource,selectfiles,[('subject_id','subject_id')]),
                 #(infosource,selectfiles_segment, [('subject_id','subject_id')]),
            (selectfiles,merge,[('func','in_files')]),
            (realign,meanfunc,[('modified_in_files','in_file')]),
            (selectfiles,segment,[('anat','channel_files')]),
            (meanfunc,bet_func,[('out_file','in_file')]),
            (selectfiles,bet_struct,[('anat','in_file')]),
            (bet_func,coregister,[('out_file','target')]),
            (bet_struct,coregister,[('out_file','source')]),
            (coregister,normalize_struct,[('coregistered_source','apply_to_files')]),
            #(selectfiles_segment,normalize_struct,[('forward_deformation_field','deformation_file')]),
            (realign,normalize_func,[('modified_in_files','apply_to_files')]),
            (segment,normalize_mask,[('forward_deformation_field','deformation_file')]),
            (bet_func,normalize_mask,[('mask_file','apply_to_files')]),
            (segment,normalize_func,[('forward_deformation_field','deformation_file')]),
            (normalize_func,smooth[('normalized_files','in_files')]),
            (smooth, bandpass, [('smoothed_files','in_files')]),
            (normalize_func, bandpass_unsmooth, [('normalized_files','in_files')])])
number_of_slices = 25
if slice_timing == 0:
    preproc.connect([(merge,realign,[('merged_file','in_files')])])
    preproc.base_dir = os.path.join(experiment_dir,working_dir)
else:
    interleaved_order = range(1,number_of_slices+1,2) + range(2,number_of_slices+1,2)
    slicetiming = Node(SliceTiming(num_slices=number_of_slices,time_repetition=TR,time_acquisition=TR-TR/number_of_slices,slice_order=interleaved_order,ref_slice=2),name="slicetiming")
    preproc.connect([(merge , slicetiming,[('merged_file' , 'in_files')]),(slicetiming, realign,[('timecorrected_files', 'in_files')])])
    working_dir = (working_dir + 'SliceTimed')
    output_dir = (output_dir + 'SliceTimed')
    preproc.base_dir = os.path.join(experiment_dir,working_dir)


#create datasink to save results from preprocessing
# Datasink
datasink = Node(DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_subject_id', '')]
datasink.inputs.substitutions = substitutions

# Connect SelectFiles and DataSink to the workflow
preproc.connect([(normalize_func,datasink, [('normalized_files','normalizedfunc.@files')]),
                (normalize_struct,datasink, [('normalized_files','normalizedstruct.@files')]),
                (segment,datasink,[('forward_deformation_field','segment.@deformation_field'),
                                   ('normalized_class_images','segment.@mnormalized_class_images'),
                                    ('bias_corrected_images','segment.@bias_corrected_images')]),
                (smooth, datasink, [('smoothed_files','smooth')]),
                (bandpass, datasink, [('out_files','filtered')]),
                (bandpass_unsmooth, datasink, [('out_files','filtered_unsmooth')]),
                (realign, datasink, [('realignment_parameters','realignment_parameters')]),
                (normalize_mask, datasink, [('normalized_files','epimask')])])

preproc.write_graph(graph2use='colored')
preproc.run('MultiProc', plugin_args={'n_procs':20}) #, plugin_args= {'qsub_args':'-q short.q', 'dont_resubmit_completed_jobs':True})



