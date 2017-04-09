# Python Notebooks for MXNet

This repo contains various notebooks ranging from basic usages of MXNet to
state-of-the-art deep learning applications.

## Outline

### Tutorials

*  MNIST: Recognize handwritten digits with multilayer perceptrons and convolutional neural networks
* Recognize image objects with pre-trained model on the full Imagenet dataset that containing more than 10M images and over 10K classes
*  Char-LSTM: Generates Obama's speeches with character-level LSTM.
* Matrix Factorization: Recommend movies to users.

### Basic Concepts

* NDArray: manipulating multi-dimensional array
* Symbol: symbolic expression for neural networks
* Module : intermediate-level and high-level interface for neural network training and inference.
* Loading data : feeding data into training/inference programs
* Mixed programming: developing training algorithms by using NDArray and Symbol together.

### How Tos
* Use a pretrainde model for fine-tune
* Use a pretrained 50 layers' [deep residual learning](https://arxiv.org/abs/1512.03385)(resnet) model for prediction and feature extraction
* Use a pretrained Inception-BatchNorm Network.


## How to use

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

Some general guidelines:

- A notebook covers a single concept or application
- Try to be as basic as possible. Put advanced usages at the end, and allow reader to skip it.
- Keep the cell outputs on the notebooks so that readers can see the results without running
