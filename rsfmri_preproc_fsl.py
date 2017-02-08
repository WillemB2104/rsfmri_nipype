"""
rs-fMRI pre-processing:
First draft for a resting-state pre-processing pipeline build with Nipype, using FSL (FEAT/MELODIC) defaults.

This pipeline involves the following:

- Motion correction (MCFLIRT),
- Slice timing correction (optional),
- Spatial smoothing (susan),
- Intensity normalization (single scaling factor, "grand mean scaling")
- Temporal highpass filtering (optional),
- Registration:
    - FLIRT (bbr) used for rigid transformation from functional image to high resolution anatomical
    - FLIRT (12 degrees of freedom) affine transformation used for anatomical to target template
    - FNIRT used for nonlinear transformation from anatomical to target template
- Normalization:
    - of structural images to original & downs-sampled template using spline interpolation
    - of functional images to down-sampled template using trilinear interpolation

Created by Willem Bruin
"""

import os
from os import path as osp
from time import time

import nipype.interfaces.fsl as fsl
from nipype import LooseVersion
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.utility import IdentityInterface, Merge
from nipype.pipeline.engine import Workflow, Node, MapNode

t_start = time()

# Check FSL version
version = 0
if fsl.Info.version() and LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
    version = 507

# Set up interface defaults
MatlabCommand.set_default_paths('/usr/local/matlabtools/2016a/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Disabled default output type as this will force all other file extensions to NIFTI as well
# fsl.FSLCommand.set_default_output_type('NIFTI')

BASE_DIR = '/data/wbbruin/Desktop/resting-state_nipype/testset'
DATA_DIR = osp.join(BASE_DIR, 'data')
WORKING_DIR = osp.join(BASE_DIR, 'workdir')

# Configure pre processing parameters
template = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
template_brain = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
template_mask = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask_dil.nii.gz')

# Data dependent parameters
TR = 2.0
number_of_slices = 37
number_of_volumes = 238

susan_smooth_fwhm = 4  # FWHM of smoothing, in mm, gets converted using sqrt(8*log(2))
highpass_sigma = 100 / (2 * TR)  # Default is 100 seconds
down_sampling = 4  # Isotropic resampling used to down sample template for registration

slice_timing = True
slice_timing_interleaved = False
slice_timing_direction = 3  # direction of slice acquisition (x=1, y=2, z=3) - default is z
slice_timing_reversed_order = False  # reversed slice indexing

highpass_filtering = True

# Specify subjects used for analysis
subject_list = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# Infosource - function free node to iterate over the list of subject names (and/or sessions)
infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - uses glob and regex to find your files
templates = dict(struct='{subject_id}/structural/structural.nii.gz',
                 func='{subject_id}/functional/*.nii.gz')
selectfiles = Node(SelectFiles(templates), "selectfiles")
selectfiles.inputs.base_directory = DATA_DIR

# Node which might come in handy when piping data to interfaces that are incompatible with gzipped format
gunzip_struct = Node(Gunzip(), name="gunzip_struct")

# Reorient images to match approximate orientation of the standard template images (MNI152)
reorient_func = Node(fsl.Reorient2Std(output_type='NIFTI_GZ'), name='reorient_func')
reorient_struct = Node(fsl.Reorient2Std(output_type='NIFTI_GZ'), name='reorient_struct')

# Convert functional images to float representation (FLOAT32)
img2float = Node(fsl.ImageMaths(out_data_type='float', op_string='', suffix='_dtype'), name='img2float')


# Return the volume index of a file
def select_volume(filename, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filename).shape[3] / 2))
    else:
        raise Exception('unknown value for volume selection : %s' % which)
    return idx


# Motion correction: realign a time series to the middle volume using spline interpolation, this workflow uses MCFLIRT
# to realign the time series and ApplyWarp to apply the rigid body transformations using spline interpolation
# (unknown order).
realign_flow = Workflow(name='realign_flow')
realign_inputnode = Node(IdentityInterface(fields=['func']), name='inputspec')
realign_outputnode = Node(IdentityInterface(fields=['realigned_file',
                                                    'transformation_matrices',
                                                    'motion_parameters',
                                                    'displacement_parameters',
                                                    'motion_plots']),
                          name='outputspec')

realigner = Node(fsl.MCFLIRT(save_mats=True, stats_imgs=True, save_plots=True), name='realigner')
splitter = Node(fsl.Split(dimension='t'), name='splitter')
warper = MapNode(fsl.ApplyWarp(interp='spline'), iterfield=['in_file', 'premat'], name='warper')
joiner = Node(fsl.Merge(dimension='t'), name='joiner')
plot_motion = Node(fsl.PlotMotionParams(in_source='fsl'), name='plot_motion')
plot_motion.iterables = ('plot_type', ['rotations', 'translations'])

realign_flow.connect(realign_inputnode, 'func', realigner, 'in_file')
realign_flow.connect(realign_inputnode, ('func', select_volume, 'middle'), realigner, 'ref_vol')
realign_flow.connect(realigner, 'out_file', splitter, 'in_file')
realign_flow.connect(realigner, 'mat_file', warper, 'premat')
realign_flow.connect(realigner, 'variance_img', warper, 'ref_file')
realign_flow.connect(realigner, 'par_file', plot_motion, 'in_file')
realign_flow.connect(splitter, 'out_files', warper, 'in_file')
realign_flow.connect(warper, 'out_file', joiner, 'in_files')
realign_flow.connect(joiner, 'merged_file', realign_outputnode, 'realigned_file')
realign_flow.connect(realigner, 'mat_file', realign_outputnode, 'transformation_matrices')
realign_flow.connect(realigner, 'par_file', realign_outputnode, 'motion_parameters')
realign_flow.connect(realigner, 'rms_files', realign_outputnode, 'displacement_parameters')
realign_flow.connect(plot_motion, 'out_file', realign_outputnode, 'motion_plots')

# Correct for slice wise acquisition using FSL's SliceTimer (optional)
slicetimer = Node(fsl.SliceTimer(time_repetition=TR, interleaved=slice_timing_interleaved,
                                 slice_direction=slice_timing_direction, index_dir=slice_timing_reversed_order),
                  name='slicetimer')

# Create mean image for functional MRI data
mean_func = Node(fsl.ImageMaths(op_string='-Tmean', suffix='_mean'), name='mean_func')

# Brain extraction for structural data
bet_struct = Node(fsl.BET(), name='bet_struct')

# Strip the skull from the mean functional to generate a mask
mean_func_brain = Node(fsl.BET(mask=True, robust=True, frac=0.3), name='mean_func_brain')

# Mask functional data with skull stripped mean functional
masker_func = Node(fsl.ImageMaths(suffix='_bet', op_string='-mas'), name='masker_func')

# Determine the 2nd and 98th percentile intensities
getthresh = Node(fsl.ImageStats(op_string='-p 2 -p 98'), name='getthreshold')

# Threshold the functional data at 10% of the 98th percentile
threshold = Node(fsl.ImageMaths(out_data_type='char', suffix='_thresh'), name='threshold')


# Define a function which creates an operand string that selects 10% of the intensity
def getthreshop(thresh):
    return '-thr %.10f -Tmin -bin' % (0.1 * thresh[1])


# Determine the median value of functional data using the mask
medianval = Node(fsl.ImageStats(op_string='-k %s -p 50'), name='medianval')

# Dilate the mask
dilatemask = Node(fsl.ImageMaths(suffix='_dil', op_string='-dilF'), name='dilatemask')

# Mask the motion corrected functional data with the dilated mask
masker_func2 = Node(interface=fsl.ImageMaths(suffix='_mask', op_string='-mas'), name='maskfunc2')

# Determine the mean image for the motion corrected and (dilated) masked functional data
mean_func2 = Node(fsl.ImageMaths(op_string='-Tmean', suffix='_mean'), name='meanfunc2')

# Smooth data using SUSAN with the brightness threshold set to 75% of the median intensity value of functional data
# and a mask constituting the mean functional.

susan_flow = Workflow(name='susan_smooth')
smooth_inputnode = Node(IdentityInterface(fields=['in_file',
                                                  'median',
                                                  'mean_func']),
                        name='inputspec')

susan = Node(fsl.SUSAN(), name='susan_smooth')
susan.inputs.fwhm = susan_smooth_fwhm
susan_flow.connect(smooth_inputnode, 'in_file', susan, 'in_file')


# Determine 75% of the given median intensity value
def getbtthresh(medianval):
    return 0.75 * medianval


susan_flow.connect(smooth_inputnode, ('median', getbtthresh), susan, 'brightness_threshold')

# Merge the median values with the mean functional images into a coupled list
merge = Node(Merge(2), name='merge')

susan_flow.connect(smooth_inputnode, 'mean_func', merge, 'in1')
susan_flow.connect(smooth_inputnode, 'median', merge, 'in2')


# Return smoothing area (USAN) in tuple format
def getusans(val):
    return [tuple([val[0], 0.75 * val[1]])]


susan_flow.connect(merge, ('out', getusans), susan, 'usans')

smooth_outputnode = Node(IdentityInterface(fields=['smoothed_file']), name='outputspec')
susan_flow.connect(susan, 'smoothed_file', smooth_outputnode, 'smoothed_file')

# Mask the smoothed data with the dilated mask
masker_func3 = Node(fsl.ImageMaths(suffix='_mask', op_string='-mas'), name='maskfunc3')

# Scale the median value of the functional data to 10000
meanscale = Node(fsl.ImageMaths(suffix='_gms'), name='meanscale')


# Define a function to get the scaling factor for intensity normalization
def getmeanscale(medianvals):
    return '-mul %.10f' % (10000. / medianvals)


# Perform temporal highpass filtering on the data (optional)
highpass_flow = Workflow(name='highpass_flow')

highpass_inputnode = Node(IdentityInterface(fields=['mean_scaled_func']), name='inputspec')

highpass = Node(fsl.ImageMaths(suffix='_tempfilt'), name='highpass')

# Define a function which creates an operand string for highpass filtering
highpass_operand = lambda x: '-bptf %.10f -1' % x

# Create a node to distribute the highpass sigma value to operand function
highpass_ = Node(IdentityInterface(fields=['highpass']), name='identitynode')
highpass_.inputs.highpass = highpass_sigma

highpass_flow.connect(highpass_, ('highpass', highpass_operand), highpass, 'op_string')
highpass_flow.connect(highpass_inputnode, 'mean_scaled_func', highpass, 'in_file')

highpass_outputnode = Node(IdentityInterface(fields=['highpassed_file']), name='outputspec')

if version < 507:
    highpass_flow.connect(highpass, 'out_file', highpass_outputnode, 'highpassed_file')
else:
    # Add back the mean removed by the highpass filter operation as of FSL 5.0.7
    meanfunc4 = Node(fsl.ImageMaths(op_string='-Tmean', suffix='_mean'), name='meanfunc4')
    highpass_flow.connect(highpass_inputnode, 'mean_scaled_func', meanfunc4, 'in_file')
    addmean = Node(fsl.BinaryMaths(operation='add'), name='addmean')
    highpass_flow.connect(highpass, 'out_file', addmean, 'in_file')
    highpass_flow.connect(meanfunc4, 'out_file', addmean, 'operand_file')
    highpass_flow.connect(addmean, 'out_file', highpass_outputnode, 'highpassed_file')

# Generate a mean functional image from filtered functional data
mean_func3 = Node(fsl.ImageMaths(op_string='-Tmean', suffix='_mean'), name='meanfunc3')

# Registration workflow following FSL's defaults
register_flow = Workflow(name='register_flow')

reg_inputnode = Node(IdentityInterface(fields=['functional_image',
                                               'anatomical_image',
                                               'target_image',
                                               'target_image_brain',
                                               'target_mask',
                                               'config_file']),
                     name='inputspec')

reg_outputnode = Node(IdentityInterface(fields=['func2highres',
                                                'highres2standard',
                                                'func2standard',
                                                'highres2standard_warp',
                                                'func2standard_warp',
                                                'transformed_mean',
                                                'example_func'
                                                ]),
                      name='outputspec')

# Extract the middle (median) volume of the functional data as reference for registration
# (corresponding to FSL defaults)
extract_ref = Node(fsl.ExtractROI(t_size=1), name='extract_ref')
register_flow.connect(reg_inputnode, 'functional_image', extract_ref, 'in_file')
register_flow.connect(reg_inputnode, ('functional_image', select_volume, 'middle'), extract_ref, 't_min')

# Estimate the tissue classes from the anatomical image.
stripper = Node(fsl.BET(), name='stripper')
register_flow.connect(reg_inputnode, 'anatomical_image', stripper, 'in_file')
fast = Node(fsl.FAST(), name='fast')
register_flow.connect(stripper, 'out_file', fast, 'in_files')

# Binarize the segmentation
binarize = Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'), name='binarize')
pickindex = lambda x, i: x[i]
register_flow.connect(fast, ('partial_volume_files', pickindex, 2), binarize, 'in_file')

# Calculate rigid transform from the reference functional image to its anatomical image.
func2highres = Node(fsl.FLIRT(), name='func2highres')
func2highres.inputs.dof = 6
register_flow.connect(extract_ref, 'roi_file', func2highres, 'in_file')
register_flow.connect(stripper, 'out_file', func2highres, 'reference')

# Now use bbr cost function to improve the transform
func2highres_bbr = Node(fsl.FLIRT(), name='func2highres_bbr')
func2highres_bbr.inputs.dof = 6
func2highres_bbr.inputs.cost = 'bbr'
func2highres_bbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch')
register_flow.connect(extract_ref, 'roi_file', func2highres_bbr, 'in_file')
register_flow.connect(binarize, 'out_file', func2highres_bbr, 'wm_seg')
register_flow.connect(reg_inputnode, 'anatomical_image', func2highres_bbr, 'reference')
register_flow.connect(func2highres, 'out_matrix_file', func2highres_bbr, 'in_matrix_file')

# Calculate affine transform from anatomical to target (default interpolation is trilinear)
highres2standard = Node(fsl.FLIRT(), name='highres2standard_affine')
highres2standard.inputs.dof = 12
highres2standard.inputs.searchr_x = [-90, 90]
highres2standard.inputs.searchr_y = [-90, 90]
highres2standard.inputs.searchr_z = [-90, 90]
register_flow.connect(stripper, 'out_file', highres2standard, 'in_file')
register_flow.connect(reg_inputnode, 'target_image_brain', highres2standard, 'reference')

# Calculate nonlinear transform from anatomical to target (default warp resolution is 10,10,10)
highres2standard_warp = Node(fsl.FNIRT(), name='highres2standard_nonlinear')
highres2standard_warp.inputs.fieldcoeff_file = True
register_flow.connect(highres2standard, 'out_matrix_file', highres2standard_warp, 'affine_file')
register_flow.connect(reg_inputnode, 'anatomical_image', highres2standard_warp, 'in_file')
register_flow.connect(reg_inputnode, 'config_file', highres2standard_warp, 'config_file')
register_flow.connect(reg_inputnode, 'target_image', highres2standard_warp, 'ref_file')
register_flow.connect(reg_inputnode, 'target_mask', highres2standard_warp, 'refmask_file')

# Combine linear FLIRT transformations into one to create a transformation matrix from functional to target
func2standard = Node(fsl.ConvertXFM(concat_xfm=True), name='func2standard_linear')
register_flow.connect(func2highres_bbr, 'out_matrix_file', func2standard, 'in_file')
register_flow.connect(highres2standard, 'out_matrix_file', func2standard, 'in_file2')

# Combine non linear FNIRT transformations into one to create a transformation matrix from functional to target
func2standard_warp = Node(fsl.ConvertWarp(), name='example_func2standard_nonlinear')
register_flow.connect(reg_inputnode, 'target_image_brain', func2standard_warp, 'reference')
register_flow.connect(func2highres_bbr, 'out_matrix_file', func2standard_warp, 'premat')
register_flow.connect(highres2standard_warp, 'fieldcoeff_file', func2standard_warp, 'warp1')

# Apply nonlinear transform to the reference image.
warpmean = Node(fsl.ApplyWarp(interp='trilinear'), name='warpmean')
register_flow.connect(extract_ref, 'roi_file', warpmean, 'in_file')
register_flow.connect(reg_inputnode, 'target_image_brain', warpmean, 'ref_file')
register_flow.connect(func2standard_warp, 'out_file', warpmean, 'field_file')

# Assign all the output files
register_flow.connect(extract_ref, 'roi_file', reg_outputnode, 'example_func')
register_flow.connect(warpmean, 'out_file', reg_outputnode, 'transformed_mean')
register_flow.connect(func2highres_bbr, 'out_matrix_file', reg_outputnode, 'func2highres')
register_flow.connect(highres2standard, 'out_matrix_file', reg_outputnode, 'highres2standard')
register_flow.connect(highres2standard_warp, 'fieldcoeff_file', reg_outputnode, 'highres2standard_warp')
register_flow.connect(func2standard, 'out_file', reg_outputnode, 'func2standard')
register_flow.connect(func2standard_warp, 'out_file', reg_outputnode, 'func2standard_warp')

# Assign templates
register_flow.inputs.inputspec.target_image = template
register_flow.inputs.inputspec.target_image_brain = template_brain
register_flow.inputs.inputspec.target_mask = template_mask

# Check registration
slicer = MapNode(fsl.Slicer(), iterfield=['in_file'], name="slicer")
slicer.inputs.image_edges = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_edges.nii.gz')
slicer.inputs.args = '-a'

# Down sample template using isotropic resampling
down_sampler_ = Node(IdentityInterface(fields=['in_file']), name='identitynode')
down_sampler_.inputs.in_file = template_brain
down_sampler = Node(fsl.FLIRT(), name='down_sampler')
down_sampler.inputs.apply_isoxfm = down_sampling

# Normalize functional images to down sampled template
warpall_func = MapNode(fsl.ApplyWarp(interp='trilinear'), iterfield=['in_file'], nested=True, name='warpall_func')

# Normalize structural images to original template
warpall_struct_org = MapNode(fsl.ApplyWarp(interp='spline'), iterfield=['in_file'], nested=True,
                             name='warpall_struct_org')

# Normalize structural images to down sampled template
warpall_struct = MapNode(fsl.ApplyWarp(interp='spline'), iterfield=['in_file'], nested=True, name='warpall_struct')

# Connect all components of the preprocessing workflow
preproc = Workflow(name="preproc")
preproc.base_dir = WORKING_DIR

preproc.connect([
    (reorient_struct, bet_struct, [('out_file', 'in_file')]),
    (reorient_func, img2float, [('out_file', 'in_file')]),
    (img2float, realign_flow, [('out_file', 'inputspec.func')]),
    (realign_flow, slicetimer, [('outputspec.realigned_file', 'in_file')]),

    (slicetimer, mean_func, [('slice_time_corrected_file', 'in_file')]),
    (mean_func, mean_func_brain, [('out_file', 'in_file')]),
    (mean_func_brain, masker_func, [('mask_file', 'in_file2')]),
    (slicetimer, masker_func, [('slice_time_corrected_file', 'in_file')]),

    (img2float, register_flow, [('out_file', 'inputspec.functional_image')]),
    (reorient_struct, register_flow, [('out_file', 'inputspec.anatomical_image')]),
    (register_flow, slicer, [('outputspec.transformed_mean', 'in_file')]),

    (masker_func, getthresh, [('out_file', 'in_file')]),
    (masker_func, threshold, [('out_file', 'in_file')]),
    (getthresh, threshold, [(('out_stat', getthreshop), 'op_string')]),
    (slicetimer, medianval, [('slice_time_corrected_file', 'in_file')]),
    (threshold, medianval, [('out_file', 'mask_file')]),
    (threshold, dilatemask, [('out_file', 'in_file')]),
    (slicetimer, masker_func2, [('slice_time_corrected_file', 'in_file')]),
    (dilatemask, masker_func2, [('out_file', 'in_file2')]),
    (masker_func2, mean_func2, [('out_file', 'in_file')]),

    (masker_func2, susan_flow, [('out_file', 'inputspec.in_file')]),
    (medianval, susan_flow, [('out_stat', 'inputspec.median')]),
    (mean_func2, susan_flow, [('out_file', 'inputspec.mean_func')]),

    (susan_flow, masker_func3, [('outputspec.smoothed_file', 'in_file')]),
    (dilatemask, masker_func3, [('out_file', 'in_file2')]),
    (masker_func3, meanscale, [('out_file', 'in_file')]),
    (medianval, meanscale, [(('out_stat', getmeanscale), 'op_string')]),

    (meanscale, highpass_flow, [('out_file', 'inputspec.mean_scaled_func')]),
    (highpass_flow, mean_func3, [('outputspec.highpassed_file', 'in_file')]),

    (down_sampler_, down_sampler, [('in_file', 'in_file')]),
    (down_sampler_, down_sampler, [('in_file', 'reference')]),

    (highpass_flow, warpall_func, [('outputspec.highpassed_file', 'in_file')]),
    (down_sampler, warpall_func, [('out_file', 'ref_file')]),
    (register_flow, warpall_func, [('outputspec.func2standard_warp', 'field_file')]),

    (bet_struct, warpall_struct, [('out_file', 'in_file')]),
    (down_sampler, warpall_struct, [('out_file', 'ref_file')]),
    (register_flow, warpall_struct, [('outputspec.highres2standard_warp', 'field_file')]),

    (bet_struct, warpall_struct_org, [('out_file', 'in_file')]),
    (down_sampler_, warpall_struct_org, [('in_file', 'ref_file')]),
    (register_flow, warpall_struct_org, [('outputspec.highres2standard_warp', 'field_file')]),
])

# Datasink
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = BASE_DIR
datasink.inputs.container = 'outputs'

report = Node(DataSink(), name="report")
report.inputs.base_directory = BASE_DIR
report.inputs.container = 'registration_report'

# Use the following DataSink output substitutions
substitutions = [('_subject_id', ''),
                 ('_session_id_', '')]
datasink.inputs.substitutions = substitutions

preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')
                                            # ,('session_id', 'session_id')
                                            ]),
                 (selectfiles, reorient_struct, [('struct', 'in_file')]),
                 (selectfiles, reorient_func, [('func', 'in_file')]),
                 (realign_flow, datasink, [('outputspec.realigned_file', 'realigned_file'),
                                           ('outputspec.transformation_matrices', 'transformation_matrices'),
                                           ('outputspec.motion_parameters', 'motion_parameters'),
                                           ('outputspec.displacement_parameters', 'displacement_parameters'),
                                           ('outputspec.motion_plots', 'motion_plots')]),
                 (dilatemask, datasink, [('out_file', 'dilated_mask')]),
                 (register_flow, datasink, [('outputspec.transformed_mean', 'transformed_mean'),
                                            ('outputspec.example_func', 'example_func'),
                                            ('outputspec.func2highres', 'func2highres'),
                                            ('outputspec.highres2standard', 'highres2standard'),
                                            ('outputspec.highres2standard_warp', 'highres2standard_warp'),
                                            ('outputspec.func2standard', 'func2standard'),
                                            ('outputspec.func2standard_warp', 'func2standard_warp'),
                                            ]),
                 (mean_func_brain, datasink, [('mask_file', 'mask')]),
                 (masker_func, datasink, [('out_file', 'masked_file')]),
                 (susan_flow, datasink, [('outputspec.smoothed_file', 'smoothed_file')]),
                 (slicer, report, [('out_file', 'registration.@registration')]),
                 (warpall_func, datasink, [('out_file', 'normalized_func')]),
                 (warpall_struct, datasink, [('out_file', 'normalized_struct')]),
                 ])


preproc.base_dir = WORKING_DIR
preproc.write_graph(graph2use='colored')
preproc.run('MultiProc', plugin_args={'n_procs': 15})

t_end = time()
print ('Time taken: {:.2f}min'.format((t_end - t_start) / 60.))
