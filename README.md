**Note: We've moved the active work on this repo to https://github.com/dmlc/mxnet/tree/master/docs. If you are looking for docs related to a new, dynamic, elegant and easy to use imperative interface for MXNet, check out http://gluon.mxnet.io/ or https://github.com/zackchase/mxnet-the-straight-dope**


# MXNet Notebooks


This repo contains various notebooks ranging from basic usages of MXNet to
state-of-the-art deep learning applications.

## How to use

### Python

The python notebooks are written in [Jupyter](http://jupyter.org/).

- **View** We can view the notebooks on either
  [github](https://github.com/dmlc/mxnet-notebooks/blob/master/python/outline.ipynb)
  or
  [nbviewer](http://nbviewer.jupyter.org/github/dmlc/mxnet-notebooks/blob/master/python/outline.ipynb). But
  note that the former may be failed to render a page, while the latter has
  delays to view the recent changes.

- **Run** We can run and modify these notebooks if both [mxnet](http://mxnet.io/get_started/index.html#setup-and-installation) and [jupyter](http://jupyter.org/) are
  installed. Here is an [example script](https://gist.github.com/mli/b64322f446b2043e3350ddcbfa5957be) to install all these packages on Ubuntu.

  If you have a AWS account, here is an easier way to run the notebooks:

  1.  Launch a g2.2xlarge or p2.2xlarge instance by using AMI `ami-fe217de9` on N. Virginia (us-east-1). This AMI is built by using  [this script](https://gist.github.com/mli/b64322f446b2043e3350ddcbfa5957be). Remember to open the TCP port 8888 in the security group.

  2.  Once launch is succeed, setup the following variable with proper value

    ```bash
      export HOSTNAME=ec2-107-22-159-132.compute-1.amazonaws.com
      export PERM=~/Downloads/my.pem
    ```

   3. Now we should be able to ssh to the machine by

      ```bash
        chmod 400 $PERM
        ssh -i $PERM -L 8888:localhost:8888 ubuntu@HOSTNAME
      ```

      Here we forward the EC2 machine's 8888 port into localhost.

   4. Clone this repo on the EC2 machine and run jupyter

      ```bash
        git clone https://github.com/dmlc/mxnet-notebooks
        jupyter notebook
      ```
   	  We can optional run `~/update_mxnet.sh` to update MXNet to the newest version.

   5. Now we are able to view and edit the notebooks on the browser using the URL: http://localhost:8888/tree/mxnet-notebooks/python/outline.ipynb


## How to develop

Some general guidelines

- A notebook covers a single concept or application
- Try to be as basic as possible. Put advanced usages at the end, and allow reader to skip it.
- Keep the cell outputs on the notebooks so that readers can see the results without running
