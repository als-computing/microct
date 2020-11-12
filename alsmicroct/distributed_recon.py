# standard library
import collections
import datetime
import json
import os
import time

# 3rd party
from IPython import display
from tqdm.auto import tqdm
import nbclient.exceptions
import papermill as pm
import yaml

from dask.distributed import Client, as_completed

class ReconstructionAnalysis(object):
    """
    ALS Tomographic Reconstruction Analysis, meant to be run from inside of a Jupyter Notebook.
    """
    
    _required_params = ['inputPath', 'filename', 'filetype']
    
    def __init__(self, template_nb=None, label=None, description=None, provenance_file=None):
        """
        Store a label, optional description, and optionally load a provenance file from a previous analysis.
        """
        if template_nb is None:
            raise TypeError("template_nb is None, must specify a valid Jupyter Notebook")
        elif not os.path.exists(template_nb):
            raise FileNotFoundError("template_nb {} does not exist, a valid Jupyter Notebook is required".format(template_nb))
        else:
            # TODO check for parameters tag
            self.template_nb = template_nb

        
        if label is None:
            self.label = datetime.datetime.now().isoformat()
        else:
            self.label = label
        
        self.description = None
        if description is not None:
            self.description = description
        
        if provenance_file is None:
            self._provenance = {}
        else:
            if not os.path.exists(provenance_file):
                raise FileNotFoundError("provenance_file {} not found".format(provenance_file))
            else:
                self._provenance_file = provenance_file
                with open(self._provenance_file, 'r') as f:
                    self._provenance = yaml.load(f)

        self._provenance['label'] = self.label
        self._provenance['description'] = self.description
        self._provenance['template_nb'] = self.template_nb

    def _validate_inputs(self, outputdir=None, params=None, timepoints=None, dask_client=None):
        """
        Check all analysis inputs for possible error conditions, and then store them for the run.
        """
        
        if outputdir is None:
            raise TypeError("outputdir is None, must specify a directory to store outputs")
        elif not os.path.exists(outputdir):
            raise FileNotFoundError("outputdir {} was not found, must specify an absolute path".format(outputdir))            
        else:
            self._outputdir = os.path.join(outputdir, self.label)
            
            # check that outputdir exists, if not then create it
            if not os.path.exists(self._outputdir):
                os.mkdir(self._outputdir)
        
        if params is None:
            raise TypeError("params is None, must provide a dictionary " + 
                            " containing keys/values for at least {}".format(self._required_params))
        elif not isinstance(params, dict):
            raise TypeError("params is not a dictionary")
        else:
            missing = []
            for p in self._required_params:
                if p not in params:
                    missing.append(p)

            if len(missing) > 0:
                raise KeyError("params dictionary must contain keys/values for at least {}, missing {}".format(
                    self._required_params, missing))

            self._params = params

        if 'inputPath' in params:
            inputdir = params['inputPath']
        else:
            raise KeyError("'inputPath' not defined in params, must contain an absolute path to a valid input directory")
        
        if inputdir is None:
            raise TypeError("inputdir is None, must specify an input data source")
        elif not os.path.exists(inputdir):
            raise FileNotFoundError("Unable to find Input directory {}".format(inputdir))
        elif not os.path.isdir(inputdir):
            raise NotADirectoryError("Input directory {} is not a directory".format(inputdir))
        else:
            self._inputdir = inputdir

        if 'filename' in params:
            filename = params['filename']
        else:
            raise KeyError("'filename' not defined in params, must contain an absolute path to a valid file")
        
        if filename is None:
            raise TypeError("filename is None, must specify an input data file")
        else:
            absfilename = os.path.join(inputdir, filename)
            
            if not os.path.exists(absfilename):
                raise FileNotFoundError("Unable to find input file {}".format(absfilename))
            elif not os.path.isfile(absfilename):
                raise TypeError("Input file {} is not a file".format(absfilename))
            else:
                self._inputfile = absfilename

        if timepoints is None:
            self._timepoints = [0]
        else:
            self._timepoints = timepoints
            
        self._provenance['outputdir'] = self._outputdir
        self._provenance['params'] = self._params
        self._provenance['timepoints'] = self._timepoints


    def _save_provenance_file(self):
        # save the input parameters out to a file with the output notebooks and data
        provenance_file = os.path.join(
            self._outputdir, 
            "{}_{}.yaml".format(self._params['filename'], self.label))
        with open(provenance_file, 'w') as f:
            yaml.dump(self._provenance, stream=f)


    def _submit_tasks(self, dask_client=None):
        _params = dict(**self._params)
        
        # submit all Papermill tasks to Dask
        submits = []
        with tqdm(total=len(self._timepoints), desc="Tasks submitted", unit="task") as submits_pbar:
            for timepoint in self._timepoints:
                _params['timepoint'] = timepoint
                out_nb = os.path.join(
                    self._outputdir, 
                    '{}_{}.ipynb'.format(self.label, timepoint))

                submits.append(
                    dask_client.submit(
                        pm.execute_notebook, 
                        self.template_nb, 
                        out_nb, 
                        _params,
                        start_timeout=60,
                        progress_bar=False))
                submits_pbar.update(1)
                time.sleep(1)

        print("{}: {} tasks submitted, {} timepoints".format(self.label, len(submits), len(self._timepoints)))

        last_exc = None
        completed = []
        failed = []
        # wait for all the tasks to complete
        with tqdm(total=len(submits), desc="Tasks completed", unit="task") as completed_pbar:
            for future in as_completed(submits):
                try:
                    x = future.result()
                    completed.append(x)
                except nbclient.exceptions.DeadKernelError as e:
                    timepoint = submits.index(future)
                    _params['timepoint'] = timepoint
                    failed.append(_params)
                except Exception as e:
                    print(e)
                    timepoint = submits.index(future)
                    _params['timepoint'] = timepoint
                    failed.append(_params)
                finally:
                    completed_pbar.update(1)
        
        return completed, failed
    
    
    def _resubmit_tasks(self, failed=None, dask_client=None):
        # cleanup any remaining tasks, if present
        resubmits = []
        completed = []
        failed = failed
        while len(failed) > 0:
            display.clear_output(wait=True)

            with tqdm(total=len(failed), desc="Tasks resubmitted", unit="task") as resubmits_pbar:
                for task_params in failed:
                    out_nb = os.path.join(
                        self._outputdir, 
                        '{}_{}.ipynb'.format(self.label, task_params['timepoint']))

                    resubmits.append(
                        dask_client.submit(
                            pm.execute_notebook, 
                            self.template_nb, 
                            out_nb, 
                            task_params, 
                            progress_bar=False))
                    resubmits_pbar.update(1)
                    time.sleep(1)

            with tqdm(total=len(resubmits), desc="Cleanup Tasks completed", unit="task") as cleanup_pbar:
                for future in as_completed(resubmits):
                    try:
                        x = future.result()

                        # clear the failure
                        i = resubmits.index(future)
                        failed[i] = None
                        resubmits[i] = None
                        completed.append(x)
                    except RuntimeError as e:
                        last_exc = e    
                    finally:
                        cleanup_pbar.update(1)

            failed = [f for f in failed if f is not None]
            resubmits = [r for r in resubmits if r is not None]
        
        return completed, failed

    
    def run_analysis(self, outputdir=None, params=None, timepoints=None, dask_client=None):
        """
        Run a tomopy anaylsis with Papermill and Dask.
        """
        
        self._validate_inputs(
            outputdir=outputdir,
            params=params,
            timepoints=timepoints,
            dask_client=dask_client
        )
        
        imagesdir = os.path.join(self._outputdir, "images")

        # create the output directory as needed
        if not os.path.exists(self._outputdir):
            os.mkdir(self._outputdir)

        # create the images subdir
        if not os.path.exists(imagesdir):
            os.mkdir(imagesdir)

        self._params['fulloutputPath'] = imagesdir

        self._save_provenance_file()
        
        print("Running all timepoints for {}".format(self.label))
        
        completed, failed = self._submit_tasks(dask_client)
        
        if len(failed) > 0:
            completed_new, failed = self._resubmit_tasks(failed, dask_client)
            
            if len(completed_new) > 0:
                completed.extend(completed_new)        
        
        print("{} runs completed for {} without error.".format(len(completed), self.label))
        
        if len(failed) > 0:
            print("Runs that failed for {} even after resubmit:".format(self.label))
            for fail in failed:
                print(failed)
        
        print("Output notebooks and data for {} in:\n {}\n\n".format(self.label, self._outputdir))

        num_notebooks = 0
        for entry in os.scandir(self._outputdir):
            if entry.name.endswith(".ipynb"):
                num_notebooks += 1

        num_images = 0
        last_index = 0
        for entry in os.scandir(imagesdir):
            if entry.name.endswith(".tiff"):
                num_images += 1

                current_index_string = entry.name.split("_")[-1].split('.')[0]

                if '-' in current_index_string:
                    current_index_string = current_index_string.split('-')[0]

                current_index = int(current_index_string)

                if last_index < current_index:
                    last_index = current_index
        
        print("{} Jupyter Notebooks were created for {}".format(num_notebooks, self.label))
        print("{} Images were created for {}, with last_index: {}\n\n\n".format(num_images, self.label, last_index))
        
        return completed, failed