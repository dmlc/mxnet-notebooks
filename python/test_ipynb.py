"""
    This script runs notebooks in selected directory and report
    errors for each notebook.
    
    Traceback information can be found in the output notebooks
    generated in coresponding output directories.
    
    Before running this scripe, make sure all the notebooks have
    been run at least once and outputs are generated.
"""

import os
import errno
import json
import ConfigParser
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
        
        Parameters
        ----------
        path : str
        notebook file path.
        
        Returns
        -------
        error : str
        notebook first cell execution errors.
    """
    error = ""
    parent_dir, nb_name = os.path.split(path)
    with open(path) as nb_file:
        nb = nbformat.read(nb_file, as_version=4)
        ep = ExecutePreprocessor(timeout=900, kernel_name='python2')
        #Use a loop to avoid "Kernel died before replying to kernel_info" error, repeat 5 times
        for _ in range(0, 5):
            error = ""
            try:
                ep.preprocess(nb, {'metadata': {'path': parent_dir}})
            except Exception as e:
                error = str(e)
            finally:
                if error != 'Kernel died before replying to kernel_info':
                    output_dir = parent_dir + "/test_output"
                    output_nb = output_dir + "/" + os.path.splitext(nb_name)[0] + "_output.ipynb"
                    #Trap an EEXIST to avoid race condition
                    try:
                        os.makedirs(output_dir)
                    except OSError as exception:
                        if exception.errno != errno.EEXIST:
                            raise
                    with open(output_nb, mode='w') as f:
                        nbformat.write(nb, f)
                    f.close()
                    nb_file.close()
                    if len(error) == 0:
                        cell_num = _verify_output(path, output_nb)
                        if cell_num > 0:
                            error = "Output in cell No.%d has changed." % cell_num
                    return error
    return error


def _verify_output(origin_nb, output_nb):
    """Compare the output cells of testing output notebook with original notebook.

        Parameters
        ----------
        origin_nb : str
        original notebook file path.
        
        output_nb : str
        output notebook file path.
        
        Returns
        -------
        cell_num : int
        First cell number in which outputs are incompatible
    """
    cell_num = 0
    origin_nb_file = open(origin_nb)
    origin_nb_js = json.load(origin_nb_file)
    output_nb_file = open(output_nb)
    output_nb_js = json.load(output_nb_file)
    for origin_cell, output_cell in zip(origin_nb_js["cells"], output_nb_js["cells"]):
        if len(origin_cell["source"]) == 0 or origin_cell["source"][0] == "# Output may vary\n" or not origin_cell.has_key("outputs"):
            continue
        if _extract_output(origin_cell["outputs"]) != _extract_output(output_cell["outputs"]):
            cell_num = origin_cell["execution_count"]
            break
    origin_nb_file.close()
    output_nb_file.close()
    return cell_num


def _extract_output(outputs):
    """Extract text part of ouput of a notebook cell.
        
        Parasmeters
        -----------
        outputs : list
        list of output
        
        Returns
        -------
        ret : str
        Concatenation of all text output contents
    """
    ret = ''
    for dict in outputs:
        for key, val in dict.items():
            if str(key).startswith('text'):
                for content in val:
                    ret += str(content)
            elif key == 'data':
                for dt_key, dt_val in val.items():
                    if str(dt_key).startswith('text'):
                        for dt_content in dt_val:
                            if not str(dt_content).startswith('<matplotlib') and not str(dt_content).startswith('<graphviz'):
                                ret += str(dt_content)
    return ret
                

configParser = ConfigParser.RawConfigParser()
configFilePath = 'test_config.txt'
configParser.read(configFilePath)
test_dirs = configParser.get('Folder Path', 'path').split(', ')
failed_notebooks = []
total_num = 0
fail_num = 0
succ_num = 0
for dir in test_dirs:
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.ipynb') and not file.endswith('-checkpoint.ipynb') and not file.endswith('_output.ipynb'):
                notebook = os.path.join(root, file)
                parent_dir = os.path.dirname(notebook)
                if parent_dir == "output":
                    continue
                print "Start to test %s.\n" % notebook
                error = _notebook_run(notebook)
                if len(error) == 0:
                    succ_num += 1
                    print "Tests for %s all passed!\n" % file
                else:
                    fail_num += 1
                    failed_notebooks.append(notebook)
                    print "Tests for %s failed:\n" % file
                    print error + '\n'
                    if (error == 'Cell execution timed out, see log for details.' or 
                        error == 'Kernel died before replying to kernel_info'):
                        print "Please manually run this notebook to debug.\n"
                    else:
                        print "See output notebook for the traceback.\n"
                total_num += 1
print "%d notebooks tested, %d succeeded, %d failed" % (total_num, succ_num, fail_num)
if len(failed_notebooks) > 0:
    print "Following are failed notebooks:"
    for nb in failed_notebooks:
        print nb