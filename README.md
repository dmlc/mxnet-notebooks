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

- **Edit** We can edit these notebooks if both mxnet and jupyter are
installed.

We show the instructions for serving the notebooks on AWS EC2.

    1. Launch a g2 or p2 instance by using AMI `ami-fe217de9` on N. Virginia
       (us-east-1). This AMI is built by using
       [this script](https://gist.github.com/mli/b64322f446b2043e3350ddcbfa5957be). Remember
       to open the TCP port 8888 in the security group.

    2. Once launch is succeed, setup the following variable with proper value

    ```bash
    export HOSTNAME=ec2-107-22-159-132.compute-1.amazonaws.com
    export PERM=~/Downloads/my.pem
    ```

    3. Now we should be able to ssh to the machine by

    ```bash
    chmod 400 $PERM
    ssh -i $PERM -L 8888:localhost:8888 ubuntu@HOSTNAME
    ```

    Here we forward the EC2 machine's port 8888 into localhost.

    4. Clone this repo on the EC2 machine and run jupyter

    ```bash
    ubuntu@ip-172-31-3-29:~$ git clone https://github.com/dmlc/mxnet-notebooks
    ubuntu@ip-172-31-3-29:~$ jupyter notebook
    ```

    We can optional run `~/update_mxnet.sh` to update MXNet to the newest
    version.

    5. Now we are able to view and edit the notebooks on the browser using the
    URL: http://localhost:8888/tree/mxnet-notebooks/python/outline.ipynb


## How to develope

Some general guidelines

- A notebook covers a single concept or application
- Try to be as basic as possible. Put advanced usages at the end, and allow reader to skip it.
- Keep the cell outputs on the notebooks so that readers can see the results without running
- Organize frequenlty asked questions on the [mxnet's issue](https://github.com/dmlc/mxnet/issues) into notebooks.
