# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:57:00 2016

@author: eaxfjord
"""



from nipype.interfaces.utility import IdentityInterface, Function, Rename
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
import nipype.interfaces.fsl as fsl
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


config.set('execution', 'remove_unnecessary_outputs', 'False')
config.set('execution', 'single_thread_matlab', 'True')
config.set('execution', 'try_hard_link_datasink', 'False')


# Specification to MATLAB
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

fsl.FSLCommand.set_default_output_type('NIFTI')

# Specify Variables
experiment_dir = '/data/eaxfjord/fmriRSWorkingDir/nipype/'
data_dir = os.path.join(experiment_dir, 'data/')  # location of the data
working_dir = 'working_dir_FirstLevel_aCompCor_TotalAmygdala_Final_PCAFIX2'
output_dir = 'output_dir_FirstLevel_aCompCor_TotalAmygdala_Final_PCAFIX2'

# Ventricle mask
VentricleMask = '/data/eaxfjord/fmriRSWorkingDir/nipype/data/masks/ventricles/ventricles.nii'

#subject_list = os.listdir(os.path.join(data_dir, 'filtered')) #all subjects

subject_list = ['_pat011_T1', '_pat011_T2']

# specify excluded subjects and remove them from the list
excluded = ['con014', 'con001','con010','con005','pat013','pat012', 'pat007','pat002']
for each in excluded:
    rm = [s for s in subject_list if each in s]
    for item in rm:
        subject_list.remove(item)


TR = 2.0
detrend_poly = 1 # remove linear trend


# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]


# SelectFiles to select data from output folder
templates={"func" : "normalizedfunc/{subject_id}/*.nii*",
            "func_filt": "filtered/{subject_id}/*.nii*",
           "anat": "normalizedstruct/{subject_id}/*.nii*",
           "func_unsmooth": "filtered_unsmooth/{subject_id}/*.nii*",
           "rp_parameters": "realignment_parameters/{subject_id}/rp*.txt",
           "epimasks" : "epimask/{subject_id}/*.nii*",
            "tissue_masks" : "segment/{subject_id}/wc*.nii*",
            "deformation_field" : "segment/{subject_id}/y_*.nii*"}
selectfiles = Node(SelectFiles(templates), "selectfiles")
selectfiles.inputs.base_directory = data_dir

#Function to build motion regressors \ derivatives and squared
def motion_regressors(motion_params, order=0, derivatives=1):

    """Compute motion regressors upto given order and derivative

    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    import os
    import numpy as np
    out_files = []
    params = np.genfromtxt(motion_params)
    out_params = params
    for d in range(1, derivatives + 1):
        cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                             params))
        out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
    out_params2 = out_params
    for i in range(2, order + 1):
        out_params2 = np.hstack((out_params2, np.power(out_params, i)))
    filename = os.path.join(os.getcwd(), "motion_regressors.txt")
    np.savetxt(filename, out_params2, fmt="%.10f")
    out_files.append(filename)
    out_file = out_files[0]

    return out_file

buildmotion = Node(Function(input_names=['motion_params', 'order' , 'derivatives'],
                                        output_names= 'out_files',function=motion_regressors,
                                        imports=imports),
                                        name='build_motion_regressors')
buildmotion.inputs.order= 0
buildmotion.inputs.derivatives=1

#bandpass/filter motion regressors, according to Hallquist 2013 et al
def bandpassMotion(in_file,lp,hp,TR):
    import os
    import numpy as np
    from nitime.timeseries import TimeSeries
    from nitime.analysis import FilterAnalyzer

    rp_data = np.genfromtxt(in_file)
    T = TimeSeries(rp_data.T,sampling_interval= TR)
    F = FilterAnalyzer(T,ub=hp, lb=lp)

    filteredRP = F.filtered_fourier.data.T
    out_file = os.path.join(os.getcwd(),'filtered_rp.txt')
    np.savetxt(out_file,filteredRP,fmt='%f')
    return out_file

bandpassMotion = Node(Function(input_names=['in_file','lp','hp','TR'],
                               output_names='out_files',
                               function= bandpassMotion),name='bandpassMotion')

bandpassMotion.inputs.lp= 0.0078
bandpassMotion.inputs.hp= 0.1
bandpassMotion.inputs.TR = TR

def aCompCor(class_images,VentriclesMask,epimask, in_file):
    
    import os
    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_to_img
    from skimage.morphology import binary_erosion
    from nipype.utils.filemanip import list_to_filename
    from sklearn.decomposition import PCA
    
    CSF = nb.load(list_to_filename(class_images[2]))
    WM = nb.load(list_to_filename(class_images[1]))
    img = nb.load(list_to_filename(in_file))
    img2data = img.get_data()[:,:,:,0]
    img2 = nb.Nifti1Image(img2data,img.get_affine(),img.get_header())
    maskepi = nb.load(epimask)

    CSFdata = CSF.get_data() >= 0.95
    # constrict to ventricle maske
    Ventricles =nb.load(VentriclesMask)
    Ventdata = Ventricles.get_data() == 100
    CSFdata = CSFdata & Ventdata

    WMdata = WM.get_data() >= 0.95
    #erode WMdata to get rid of connectionless pixel
    WMdata = binary_erosion(WMdata)
    CSF = nb.Nifti1Image(CSFdata,CSF.get_affine(),CSF.get_header())
    nb.save(CSF,'CSF_Ventricles.nii')
    WM = nb.Nifti1Image(WMdata,WM.get_affine(),WM.get_header())
    nb.save(WM,'WM.nii')
    reslicedCSF = resample_to_img(CSF,img2)
    reslicedWM = resample_to_img(WM,img2)

    rs_csf_data = reslicedCSF.get_data()
    rs_wm_data = reslicedWM.get_data()
    maskepidata = maskepi.get_data()

    rs_csf_data = rs_csf_data & maskepidata
    rs_wm_data = rs_wm_data & maskepidata
    
    csf_ts = img.get_data()[rs_csf_data>0]
    wm_ts = img.get_data()[rs_wm_data>0]
    
    X_csf = csf_ts.T
    X_wm = wm_ts.T
    
    stdCSF = np.std(X_csf,axis=0)
    stdWM = np.std(X_wm,axis=0)
    
    mean_csf = np.mean(X_csf,axis=1)
    mean_wm = np.mean(X_wm,axis=1)
    
    X_csf = (X_csf - np.mean(X_csf,axis=0))/stdCSF
    X_wm = (X_wm - np.mean(X_wm,axis=0))/stdWM    
    
    pca = PCA(n_components=0.5)    
    
    components_csf = pca.fit(X_csf.T).components_
    components_wm = pca.fit(X_wm.T).components_    
    
    all_components = np.column_stack((components_csf.T,components_wm.T))    
    out_file = os.path.join(os.getcwd(),'all_components.txt')
    out_csf = os.path.join(os.getcwd(),'csf_components.txt')
    out_wm = os.path.join(os.getcwd(),'wm_components.txt')
    
    out_csf_mean = os.path.join(os.getcwd(),'mean_csf.txt')
    out_wm_mean = os.path.join(os.getcwd(), 'mean_wm.txt')    
    
    np.savetxt(out_csf,components_csf,fmt='%.10f')
    np.savetxt(out_wm,components_wm,fmt='%.10f')
    np.savetxt(out_file,all_components,fmt='%.10f')
    
    np.savetxt(out_csf_mean, mean_csf, fmt='%.10f')
    np.savetxt(out_wm_mean, mean_wm, fmt='%.10f')
    

    return out_file
# Node to extract noise compoments from CSF and white matter (if used)
acompcor = Node(Function(input_names=['in_file','class_images',
                                          'VentriclesMask','epimask'],
                             output_names=['out_files'],
                    function=aCompCor),name='aCompCor')

acompcor.inputs.VentriclesMask = VentricleMask

#extraact median timeseries from rois
def extractROI(in_file,hemi,ROIname,epimask):

    import os
    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_to_img
    from nipype.utils.filemanip import list_to_filename

    ROIFolder = '/data/eaxfjord/fmriRSWorkingDir/nipype/data/ROI_Total/'
    allRois = os.listdir(ROIFolder)

    numberRois = [elem for elem in allRois if (hemi + '_Amygdala') in elem]


    img = nb.load(list_to_filename(in_file))
    img2data = img.get_data()[:,:,:,0]
    epi = nb.load(epimask)
    epimaskdata = epi.get_data()
    img2 = nb.Nifti1Image(img2data,img.get_affine(),img.get_header())
    
    f = numberRois[0]
    roi = nb.load(os.path.join(ROIFolder,f))
    reslicedROI = resample_to_img(roi,img2)
    roi_data = reslicedROI.get_data()

    filtered_roi = roi_data & np.squeeze(epimaskdata.astype('bool'))
    ijk = np.nonzero(filtered_roi)

    roiTS = img.get_data()[ijk]
    medianTS = np.mean(roiTS,axis=0)

    ROIname = ROIname + '_' + hemi
    out_file_main = os.path.join(os.getcwd(),'%s_ts.txt' % ROIname)
    np.savetxt(out_file_main,medianTS, fmt ='%f')

    return out_file_main, ROIname


extractroi = Node(Function(input_names = ['in_file','hemi','ROIname',
                                          'epimask'],
                           output_names = ['main_roi_ts','ROIname'],
                            function=extractROI),name='extractroi')
extractroi.iterables = [('hemi',['R','L']),('ROIname',['Amygdala'])]

def combineRegress(motion_regressors,wmcsf_regressors,detrend_poly):
    import numpy as np
    import os
    from scipy.special import legendre


    motion = np.loadtxt(motion_regressors)
    wmcsf = np.loadtxt(wmcsf_regressors)

    regressors = np.column_stack((motion,wmcsf))

    if detrend_poly:
        timepoints = regressors.shape[0]
        X = np.empty((timepoints, 0))
        for i in range(detrend_poly):
            X = np.hstack((X, legendre(
            i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
    regressors = np.column_stack((regressors, X))

    out_file =  os.path.join(os.getcwd(),'regressors.txt')
    np.savetxt(out_file,regressors, fmt='%f')

    return out_file

combine_regressors = Node(Function(input_names=['motion_regressors',
                                                'wmcsf_regressors',
                                                'detrend_poly'],
                                    output_names='regressors',
                                    function=combineRegress),
                                    name = 'combine_regressors')
combine_regressors.inputs.detrend_poly = detrend_poly


def nuis_regress(regressors, epimask, in_file):

    import os
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import list_to_filename

    img = nb.load(list_to_filename(in_file))
    regressors = np.loadtxt(regressors)
    img_data = img.get_data()

    mask = nb.load(epimask)
    mask_data = np.squeeze(mask.get_data())

    ijk = mask_data ==1


    timeseries = img_data[ijk].T

    regressors = np.column_stack([regressors,np.ones([img_data.shape[-1],1])])

    x,sum_res,rank,s = np.linalg.lstsq(regressors,timeseries)

    timeseries_hat = np.dot(regressors,x)

    residuals = timeseries - timeseries_hat

    indexes = np.where(mask_data ==1)

    rebuilt_array = np.zeros(img_data.shape)
    rebuilt_array[indexes[0],indexes[1],indexes[2]]= residuals.T

    residuals_img = nb.Nifti1Image(rebuilt_array,img.get_affine(),img.get_header())
    out_file = os.path.join(os.getcwd(),'residuals.nii')
    nb.save(residuals_img,out_file)

    return out_file

regress = Node(Function(input_names = ['in_file',
                                            'regressors',
                                            'epimask'],
                            output_names = 'residuals',
                            function = nuis_regress),name = 'nuis_regress')

def functional_connectivity(in_file,epimask,main_roi, subject_id,main_roi_name):

    import nibabel as nb
    import numpy as np
    import os

    img = nb.load(in_file)

    epi = nb.load(epimask)
    mask = np.squeeze(epi.get_data())
    img_data = img.get_data()

    main_roi = np.loadtxt(main_roi)

    # create mask index and mask
    ijk = mask == 1
    brain_timeseries = img_data[ijk].T

    # remove mean from each timeseries
    main_roi = main_roi - np.mean(main_roi)
    brain_timeseries = brain_timeseries - np.mean(brain_timeseries,
                                                  axis=0)

    # number of timepoints
    ndim = img_data.shape[-1]

    # compute std of each timeseries
    main_roi_std = np.std(main_roi)
    brain_std = np.std(brain_timeseries,axis=0)

    # calculate fc
    fc = np.dot(brain_timeseries.T,main_roi) /ndim
    fc = (fc / brain_std) / main_roi_std

    # convert to z- values
    fc_z = (np.log(1 + fc) - np.log(1 - fc)) / 2


    indexes = np.where(mask==1)

    fc_img = np.zeros(ijk.shape)
    fc_z_img = np.zeros(ijk.shape)
    fc_img[indexes[0],indexes[1],indexes[2]] = fc
    fc_z_img[indexes[0],indexes[1],indexes[2]] = fc_z

    fc_nii = nb.Nifti1Image(fc_img,img.get_affine(),img.get_header())
    fc_z_nii = nb.Nifti1Image(fc_z_img,img.get_affine(),img.get_header())

    fc_nii.set_data_dtype('float64')
    fc_z_nii.set_data_dtype('float64')

    filename = os.path.join(os.getcwd(),(subject_id + main_roi_name +'fc.nii'))
    filename_z = os.path.join(os.getcwd(),(subject_id + main_roi_name +'fc_z.nii'))

    nb.save(fc_nii,filename)
    nb.save(fc_z_nii,filename_z)

    return filename, filename_z

fc = Node(Function(input_names= ['in_file',
                                 'epimask',
                                 'main_roi','subject_id','main_roi_name'],
                   output_names= ['filename','filename_z'],
                    function = functional_connectivity),
                    name = 'fc')

#create the workflow
firstlevel = Workflow(name='firstlevel')
firstlevel.base_dir = os.path.join(experiment_dir,working_dir)

firstlevel.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                    (selectfiles, buildmotion, [('rp_parameters', 'motion_params')]),
                    (buildmotion, bandpassMotion, [('out_files', 'in_file')]),
                    (selectfiles, acompcor, [('epimasks', 'epimask')]),
                    (selectfiles, acompcor, [('tissue_masks', 'class_images')]),
                    (selectfiles, acompcor, [('func_filt', 'in_file')]),
                    (selectfiles, extractroi, [('func_unsmooth', 'in_file')]),
                    (selectfiles, extractroi, [('epimasks', 'epimask')]),
                    (bandpassMotion, combine_regressors,
                     [('out_files', 'motion_regressors')]),
                    (acompcor, combine_regressors, [('out_files',
                                                     'wmcsf_regressors')]),
                    (selectfiles, regress, [('func_filt', 'in_file')]),
                    (combine_regressors, regress, [('regressors', 'regressors')]),
                    (selectfiles, regress, [('epimasks', 'epimask')]),
                    (regress, fc, [('residuals', 'in_file')]),
                    (selectfiles, fc, [('epimasks', 'epimask')]),
                    (extractroi, fc, [('main_roi_ts', 'main_roi')]),
                    (extractroi, fc, [('ROIname', 'main_roi_name')]),
                    (infosource, fc, [('subject_id', 'subject_id')])
                    ])


rename_z = MapNode(Rename(format_string='%(ROIname)s_%(subject_id)s_zmap',
                        keep_ext=True),
                        iterfield=['subject_id','ROIname'],
                        name='namer_z')

rename_corr = MapNode(Rename(format_string='%(ROIname)s_%(subject_id)s_corr_map',
                        keep_ext=True),
                        iterfield=['subject_id','ROIname'],
                        name='namer_corr')

#create datasink to save results from preprocessing
# Datasink
datasink = Node(DataSink(parameterization=False,
                         base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('__', '/')]
datasink.inputs.substitutions = substitutions

# Connect SelectFiles and DataSink to the workflow
firstlevel.connect([(fc, rename_corr, [('filename','in_file')]),
                    (fc, rename_z, [('filename_z', 'in_file')]),
                    (infosource, rename_z, [('subject_id', 'subject_id')]),
                    (infosource, rename_corr, [('subject_id', 'subject_id')]),
                    (extractroi, rename_z, [('ROIname','ROIname')]),
                    (extractroi, rename_corr, [('ROIname', 'ROIname')]),
                    (rename_z, datasink, [('out_file', 'Zmaps')]),
                    (rename_corr, datasink, [('out_file', 'Corrmaps')])
                    ])

firstlevel.write_graph(graph2use='colored')
firstlevel.run('MultiProc', plugin_args = {'n_procs':8})#, plugin_args= {'qsub_args':'-q short.q', 'dont_resubmit_completed_jobs':True})



