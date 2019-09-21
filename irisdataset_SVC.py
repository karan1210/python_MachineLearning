{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "irisdataset_SVC.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "V2xlNBCkE8Mx",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 102
        },
        "outputId": "e6152c17-f12b-4f83-bd12-367680f7d439"
      },
      "source": [
        "!pip install pandas"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (0.24.2)\n",
            "Requirement already satisfied: numpy>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from pandas) (1.16.4)\n",
            "Requirement already satisfied: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)\n",
            "Requirement already satisfied: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.5.3)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.5.0->pandas) (1.12.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c07QUYu1FJVJ",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 102
        },
        "outputId": "deaf74f5-d129-4224-fdf8-b17063a63f15"
      },
      "source": [
        "!pip install sklearn"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: sklearn in /usr/local/lib/python3.6/dist-packages (0.0)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from sklearn) (0.21.2)\n",
            "Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sklearn) (1.3.0)\n",
            "Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sklearn) (1.16.4)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sklearn) (0.13.2)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4jQsQ-17Fc03",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "0cb7e9bb-3915-4942-8311-bc170caf5370"
      },
      "source": [
        "\n"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Package                  Version              \n",
            "------------------------ ---------------------\n",
            "absl-py                  0.7.1                \n",
            "alabaster                0.7.12               \n",
            "albumentations           0.1.12               \n",
            "altair                   3.1.0                \n",
            "astor                    0.8.0                \n",
            "astropy                  3.0.5                \n",
            "atari-py                 0.1.15               \n",
            "atomicwrites             1.3.0                \n",
            "attrs                    19.1.0               \n",
            "audioread                2.1.8                \n",
            "autograd                 1.2                  \n",
            "Babel                    2.7.0                \n",
            "backcall                 0.1.0                \n",
            "backports.tempfile       1.0                  \n",
            "backports.weakref        1.0.post1            \n",
            "beautifulsoup4           4.6.3                \n",
            "bleach                   3.1.0                \n",
            "blis                     0.2.4                \n",
            "bokeh                    1.0.4                \n",
            "boto                     2.49.0               \n",
            "boto3                    1.9.180              \n",
            "botocore                 1.12.180             \n",
            "Bottleneck               1.2.1                \n",
            "branca                   0.3.1                \n",
            "bs4                      0.0.1                \n",
            "bz2file                  0.98                 \n",
            "cachetools               3.1.1                \n",
            "certifi                  2019.6.16            \n",
            "cffi                     1.12.3               \n",
            "chainer                  5.4.0                \n",
            "chardet                  3.0.4                \n",
            "Click                    7.0                  \n",
            "cloudpickle              0.6.1                \n",
            "cmake                    3.12.0               \n",
            "colorlover               0.3.0                \n",
            "community                1.0.0b1              \n",
            "contextlib2              0.5.5                \n",
            "convertdate              2.1.3                \n",
            "coverage                 3.7.1                \n",
            "coveralls                0.5                  \n",
            "crcmod                   1.7                  \n",
            "cufflinks                0.14.6               \n",
            "cvxopt                   1.2.3                \n",
            "cvxpy                    1.0.15               \n",
            "cycler                   0.10.0               \n",
            "cymem                    2.0.2                \n",
            "Cython                   0.29.10              \n",
            "daft                     0.0.4                \n",
            "dask                     1.1.5                \n",
            "dataclasses              0.6                  \n",
            "datascience              0.10.6               \n",
            "decorator                4.4.0                \n",
            "defusedxml               0.6.0                \n",
            "descartes                1.1.0                \n",
            "dill                     0.3.0                \n",
            "distributed              1.25.3               \n",
            "Django                   2.2.2                \n",
            "dlib                     19.16.0              \n",
            "dm-sonnet                1.33                 \n",
            "docopt                   0.6.2                \n",
            "docutils                 0.14                 \n",
            "dopamine-rl              1.0.5                \n",
            "easydict                 1.9                  \n",
            "ecos                     2.0.7.post1          \n",
            "editdistance             0.5.3                \n",
            "en-core-web-sm           2.1.0                \n",
            "entrypoints              0.3                  \n",
            "enum34                   1.1.6                \n",
            "ephem                    3.7.6.0              \n",
            "et-xmlfile               1.0.1                \n",
            "fa2                      0.3.5                \n",
            "fancyimpute              0.4.3                \n",
            "fastai                   1.0.54               \n",
            "fastcache                1.1.0                \n",
            "fastdtw                  0.3.2                \n",
            "fastprogress             0.1.21               \n",
            "fastrlock                0.4                  \n",
            "fbprophet                0.5                  \n",
            "feather-format           0.4.0                \n",
            "featuretools             0.4.1                \n",
            "filelock                 3.0.12               \n",
            "fix-yahoo-finance        0.0.22               \n",
            "Flask                    1.0.3                \n",
            "folium                   0.8.3                \n",
            "future                   0.16.0               \n",
            "gast                     0.2.2                \n",
            "GDAL                     2.2.2                \n",
            "gdown                    3.6.4                \n",
            "gensim                   3.6.0                \n",
            "geographiclib            1.49                 \n",
            "geopy                    1.17.0               \n",
            "gevent                   1.4.0                \n",
            "gin-config               0.1.4                \n",
            "glob2                    0.7                  \n",
            "google                   2.0.2                \n",
            "google-api-core          1.13.0               \n",
            "google-api-python-client 1.7.9                \n",
            "google-auth              1.4.2                \n",
            "google-auth-httplib2     0.0.3                \n",
            "google-auth-oauthlib     0.4.0                \n",
            "google-cloud-bigquery    1.14.0               \n",
            "google-cloud-core        1.0.2                \n",
            "google-cloud-datastore   1.8.0                \n",
            "google-cloud-language    1.2.0                \n",
            "google-cloud-storage     1.16.1               \n",
            "google-cloud-translate   1.5.0                \n",
            "google-colab             1.0.0                \n",
            "google-pasta             0.1.7                \n",
            "google-resumable-media   0.3.2                \n",
            "googleapis-common-protos 1.6.0                \n",
            "googledrivedownloader    0.4                  \n",
            "graph-nets               1.0.4                \n",
            "graphviz                 0.10.1               \n",
            "greenlet                 0.4.15               \n",
            "grpcio                   1.15.0               \n",
            "gspread                  3.0.1                \n",
            "gspread-dataframe        3.0.2                \n",
            "gunicorn                 19.9.0               \n",
            "gym                      0.10.11              \n",
            "h5py                     2.8.0                \n",
            "HeapDict                 1.0.0                \n",
            "holidays                 0.9.10               \n",
            "html5lib                 1.0.1                \n",
            "httpimport               0.5.16               \n",
            "httplib2                 0.11.3               \n",
            "humanize                 0.5.1                \n",
            "hyperopt                 0.1.2                \n",
            "ideep4py                 2.0.0.post3          \n",
            "idna                     2.8                  \n",
            "image                    1.5.27               \n",
            "imageio                  2.4.1                \n",
            "imagesize                1.1.0                \n",
            "imbalanced-learn         0.4.3                \n",
            "imblearn                 0.0                  \n",
            "imgaug                   0.2.9                \n",
            "importlib-metadata       0.18                 \n",
            "imutils                  0.5.2                \n",
            "inflect                  2.1.0                \n",
            "intel-openmp             2019.0               \n",
            "intervaltree             2.1.0                \n",
            "ipykernel                4.6.1                \n",
            "ipython                  5.5.0                \n",
            "ipython-genutils         0.2.0                \n",
            "ipython-sql              0.3.9                \n",
            "ipywidgets               7.4.2                \n",
            "itsdangerous             1.1.0                \n",
            "jdcal                    1.4.1                \n",
            "jedi                     0.14.0               \n",
            "jieba                    0.39                 \n",
            "Jinja2                   2.10.1               \n",
            "jmespath                 0.9.4                \n",
            "joblib                   0.13.2               \n",
            "jpeg4py                  0.1.4                \n",
            "jsonschema               2.6.0                \n",
            "jupyter                  1.0.0                \n",
            "jupyter-client           5.2.4                \n",
            "jupyter-console          6.0.0                \n",
            "jupyter-core             4.5.0                \n",
            "kaggle                   1.5.4                \n",
            "kapre                    0.1.3.1              \n",
            "Keras                    2.2.4                \n",
            "Keras-Applications       1.0.8                \n",
            "Keras-Preprocessing      1.1.0                \n",
            "keras-vis                0.4.1                \n",
            "kiwisolver               1.1.0                \n",
            "knnimpute                0.1.0                \n",
            "librosa                  0.6.3                \n",
            "lightgbm                 2.2.3                \n",
            "llvmlite                 0.29.0               \n",
            "lmdb                     0.95                 \n",
            "lucid                    0.3.8                \n",
            "lunardate                0.2.0                \n",
            "lxml                     4.2.6                \n",
            "magenta                  0.3.19               \n",
            "Markdown                 3.1.1                \n",
            "MarkupSafe               1.1.1                \n",
            "matplotlib               3.0.3                \n",
            "matplotlib-venn          0.11.5               \n",
            "mesh-tensorflow          0.0.5                \n",
            "mido                     1.2.6                \n",
            "mir-eval                 0.5                  \n",
            "missingno                0.4.1                \n",
            "mistune                  0.8.4                \n",
            "mizani                   0.5.4                \n",
            "mkl                      2019.0               \n",
            "mlxtend                  0.14.0               \n",
            "more-itertools           7.1.0                \n",
            "moviepy                  0.2.3.5              \n",
            "mpi4py                   3.0.2                \n",
            "mpmath                   1.1.0                \n",
            "msgpack                  0.5.6                \n",
            "multiprocess             0.70.8               \n",
            "multitasking             0.0.9                \n",
            "murmurhash               1.0.2                \n",
            "music21                  5.5.0                \n",
            "natsort                  5.5.0                \n",
            "nbconvert                5.5.0                \n",
            "nbformat                 4.4.0                \n",
            "networkx                 2.3                  \n",
            "nibabel                  2.3.3                \n",
            "nltk                     3.2.5                \n",
            "nose                     1.3.7                \n",
            "notebook                 5.2.2                \n",
            "np-utils                 0.5.10.0             \n",
            "numba                    0.40.1               \n",
            "numexpr                  2.6.9                \n",
            "numpy                    1.16.4               \n",
            "nvidia-ml-py3            7.352.0              \n",
            "oauth2client             4.1.3                \n",
            "oauthlib                 3.0.1                \n",
            "okgrade                  0.4.3                \n",
            "olefile                  0.46                 \n",
            "opencv-contrib-python    3.4.3.18             \n",
            "opencv-python            3.4.5.20             \n",
            "openpyxl                 2.5.9                \n",
            "osqp                     0.5.0                \n",
            "packaging                19.0                 \n",
            "palettable               3.1.1                \n",
            "pandas                   0.24.2               \n",
            "pandas-datareader        0.7.0                \n",
            "pandas-gbq               0.4.1                \n",
            "pandas-profiling         1.4.1                \n",
            "pandocfilters            1.4.2                \n",
            "parso                    0.5.0                \n",
            "pathlib                  1.0.1                \n",
            "patsy                    0.5.1                \n",
            "pexpect                  4.7.0                \n",
            "pickleshare              0.7.5                \n",
            "Pillow                   4.3.0                \n",
            "pip                      19.1.1               \n",
            "pip-tools                3.6.1                \n",
            "plac                     0.9.6                \n",
            "plotly                   3.6.1                \n",
            "plotnine                 0.5.1                \n",
            "pluggy                   0.7.1                \n",
            "portpicker               1.2.0                \n",
            "prefetch-generator       1.0.1                \n",
            "preshed                  2.0.1                \n",
            "pretty-midi              0.2.8                \n",
            "prettytable              0.7.2                \n",
            "progressbar2             3.38.0               \n",
            "prometheus-client        0.7.1                \n",
            "promise                  2.2.1                \n",
            "prompt-toolkit           1.0.16               \n",
            "protobuf                 3.7.1                \n",
            "psutil                   5.4.8                \n",
            "psycopg2                 2.7.6.1              \n",
            "ptyprocess               0.6.0                \n",
            "py                       1.8.0                \n",
            "pyarrow                  0.13.0               \n",
            "pyasn1                   0.4.5                \n",
            "pyasn1-modules           0.2.5                \n",
            "pycocotools              2.0.0                \n",
            "pycparser                2.19                 \n",
            "pydot                    1.3.0                \n",
            "pydot-ng                 2.0.0                \n",
            "pydotplus                2.0.2                \n",
            "pyemd                    0.5.1                \n",
            "pyglet                   1.3.2                \n",
            "Pygments                 2.1.3                \n",
            "pygobject                3.26.1               \n",
            "pymc3                    3.7                  \n",
            "pymongo                  3.8.0                \n",
            "pymystem3                0.2.0                \n",
            "PyOpenGL                 3.1.0                \n",
            "pyparsing                2.4.0                \n",
            "pyrsistent               0.15.2               \n",
            "pysndfile                1.3.3                \n",
            "PySocks                  1.7.0                \n",
            "pystan                   2.19.0.0             \n",
            "pytest                   3.6.4                \n",
            "python-apt               1.6.4                \n",
            "python-chess             0.23.11              \n",
            "python-dateutil          2.5.3                \n",
            "python-louvain           0.13                 \n",
            "python-rtmidi            1.3.0                \n",
            "python-slugify           3.0.2                \n",
            "python-utils             2.3.0                \n",
            "pytz                     2018.9               \n",
            "PyWavelets               1.0.3                \n",
            "PyYAML                   3.13                 \n",
            "pyzmq                    17.0.0               \n",
            "qtconsole                4.5.1                \n",
            "requests                 2.21.0               \n",
            "requests-oauthlib        1.2.0                \n",
            "resampy                  0.2.1                \n",
            "retrying                 1.3.3                \n",
            "rpy2                     2.9.5                \n",
            "rsa                      4.0                  \n",
            "s3fs                     0.2.1                \n",
            "s3transfer               0.2.1                \n",
            "scikit-image             0.15.0               \n",
            "scikit-learn             0.21.2               \n",
            "scipy                    1.3.0                \n",
            "screen-resolution-extra  0.0.0                \n",
            "scs                      2.1.0                \n",
            "seaborn                  0.9.0                \n",
            "semantic-version         2.6.0                \n",
            "Send2Trash               1.5.0                \n",
            "setuptools               41.0.1               \n",
            "setuptools-git           1.2                  \n",
            "Shapely                  1.6.4.post2          \n",
            "simplegeneric            0.8.1                \n",
            "siphash                  0.0.1                \n",
            "six                      1.12.0               \n",
            "sklearn                  0.0                  \n",
            "sklearn-pandas           1.8.0                \n",
            "smart-open               1.8.4                \n",
            "snowballstemmer          1.9.0                \n",
            "sortedcontainers         2.1.0                \n",
            "spacy                    2.1.4                \n",
            "Sphinx                   1.8.5                \n",
            "sphinxcontrib-websupport 1.1.2                \n",
            "SQLAlchemy               1.3.5                \n",
            "sqlparse                 0.3.0                \n",
            "srsly                    0.0.7                \n",
            "stable-baselines         2.2.1                \n",
            "statsmodels              0.10.0               \n",
            "sympy                    1.1.1                \n",
            "tables                   3.4.4                \n",
            "tabulate                 0.8.3                \n",
            "tblib                    1.4.0                \n",
            "tensor2tensor            1.11.0               \n",
            "tensorboard              1.14.0               \n",
            "tensorboardcolab         0.0.22               \n",
            "tensorflow               1.14.0               \n",
            "tensorflow-estimator     1.14.0               \n",
            "tensorflow-hub           0.5.0                \n",
            "tensorflow-metadata      0.13.0               \n",
            "tensorflow-probability   0.7.0                \n",
            "termcolor                1.1.0                \n",
            "terminado                0.8.2                \n",
            "testpath                 0.4.2                \n",
            "text-unidecode           1.2                  \n",
            "textblob                 0.15.3               \n",
            "textgenrnn               1.4.1                \n",
            "tfds-nightly             1.0.2.dev201906280105\n",
            "tflearn                  0.3.2                \n",
            "Theano                   1.0.4                \n",
            "thinc                    7.0.4                \n",
            "toolz                    0.9.0                \n",
            "torch                    1.1.0                \n",
            "torchsummary             1.5.1                \n",
            "torchtext                0.3.1                \n",
            "torchvision              0.3.0                \n",
            "tornado                  4.5.3                \n",
            "tqdm                     4.28.1               \n",
            "traitlets                4.3.2                \n",
            "tweepy                   3.6.0                \n",
            "typing                   3.7.4                \n",
            "tzlocal                  1.5.1                \n",
            "umap-learn               0.3.9                \n",
            "uritemplate              3.0.0                \n",
            "urllib3                  1.24.3               \n",
            "vega-datasets            0.7.0                \n",
            "wasabi                   0.2.2                \n",
            "wcwidth                  0.1.7                \n",
            "webencodings             0.5.1                \n",
            "Werkzeug                 0.15.4               \n",
            "wheel                    0.33.4               \n",
            "widgetsnbextension       3.4.2                \n",
            "wordcloud                1.5.0                \n",
            "wrapt                    1.11.2               \n",
            "xarray                   0.11.3               \n",
            "xgboost                  0.90                 \n",
            "xkit                     0.0.0                \n",
            "xlrd                     1.1.0                \n",
            "xlwt                     1.3.0                \n",
            "yellowbrick              0.9.1                \n",
            "zict                     1.0.0                \n",
            "zipp                     0.5.1                \n",
            "zmq                      0.0.0                \n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nD-FGmWPFnOJ",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "from sklearn.datasets import load_iris\n",
        "from matplotlib import pyplot as plt\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qHoeFvzBF6gn",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        },
        "outputId": "d9fe6bab-5833-4564-f8a3-c3a9600b5017"
      },
      "source": [
        "iris = load_iris()\n",
        "\n",
        "iris.feature_names"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['sepal length (cm)',\n",
              " 'sepal width (cm)',\n",
              " 'petal length (cm)',\n",
              " 'petal width (cm)']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rythRqKkGZvC",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "1a12df65-9c2c-4ee3-842c-7d05f2140866"
      },
      "source": [
        "iris.target_names"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['setosa', 'versicolor', 'virginica'], dtype='<U10')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ot66835fGyGu",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "73e3d430-3162-423b-9175-902074a7b873"
      },
      "source": [
        "df = pd.DataFrame(iris.data,columns=iris.feature_names)\n",
        "df.head()\n"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>sepal length (cm)</th>\n",
              "      <th>sepal width (cm)</th>\n",
              "      <th>petal length (cm)</th>\n",
              "      <th>petal width (cm)</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5.1</td>\n",
              "      <td>3.5</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4.9</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4.7</td>\n",
              "      <td>3.2</td>\n",
              "      <td>1.3</td>\n",
              "      <td>0.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4.6</td>\n",
              "      <td>3.1</td>\n",
              "      <td>1.5</td>\n",
              "      <td>0.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5.0</td>\n",
              "      <td>3.6</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
              "0                5.1               3.5                1.4               0.2\n",
              "1                4.9               3.0                1.4               0.2\n",
              "2                4.7               3.2                1.3               0.2\n",
              "3                4.6               3.1                1.5               0.2\n",
              "4                5.0               3.6                1.4               0.2"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A5BS5x4iG72W",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "ec10f9e0-f4d0-48fa-af81-e2d0c20faeb3"
      },
      "source": [
        "df['target'] = iris.target\n",
        "df.head()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>sepal length (cm)</th>\n",
              "      <th>sepal width (cm)</th>\n",
              "      <th>petal length (cm)</th>\n",
              "      <th>petal width (cm)</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>5.1</td>\n",
              "      <td>3.5</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4.9</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>4.7</td>\n",
              "      <td>3.2</td>\n",
              "      <td>1.3</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4.6</td>\n",
              "      <td>3.1</td>\n",
              "      <td>1.5</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5.0</td>\n",
              "      <td>3.6</td>\n",
              "      <td>1.4</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target\n",
              "0                5.1               3.5  ...               0.2       0\n",
              "1                4.9               3.0  ...               0.2       0\n",
              "2                4.7               3.2  ...               0.2       0\n",
              "3                4.6               3.1  ...               0.2       0\n",
              "4                5.0               3.6  ...               0.2       0\n",
              "\n",
              "[5 rows x 5 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VMOXbbK4HBIu",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "outputId": "4aa7afbc-9628-4a21-d7a3-4a77138422e1"
      },
      "source": [
        "df[df.target==1].head()#for type 1 \n",
        "df[df.target==2].head()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>sepal length (cm)</th>\n",
              "      <th>sepal width (cm)</th>\n",
              "      <th>petal length (cm)</th>\n",
              "      <th>petal width (cm)</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>100</th>\n",
              "      <td>6.3</td>\n",
              "      <td>3.3</td>\n",
              "      <td>6.0</td>\n",
              "      <td>2.5</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>101</th>\n",
              "      <td>5.8</td>\n",
              "      <td>2.7</td>\n",
              "      <td>5.1</td>\n",
              "      <td>1.9</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>102</th>\n",
              "      <td>7.1</td>\n",
              "      <td>3.0</td>\n",
              "      <td>5.9</td>\n",
              "      <td>2.1</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>103</th>\n",
              "      <td>6.3</td>\n",
              "      <td>2.9</td>\n",
              "      <td>5.6</td>\n",
              "      <td>1.8</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>104</th>\n",
              "      <td>6.5</td>\n",
              "      <td>3.0</td>\n",
              "      <td>5.8</td>\n",
              "      <td>2.2</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "     sepal length (cm)  sepal width (cm)  ...  petal width (cm)  target\n",
              "100                6.3               3.3  ...               2.5       2\n",
              "101                5.8               2.7  ...               1.9       2\n",
              "102                7.1               3.0  ...               2.1       2\n",
              "103                6.3               2.9  ...               1.8       2\n",
              "104                6.5               3.0  ...               2.2       2\n",
              "\n",
              "[5 rows x 5 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kd62OcgAHMne",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])\n",
        "df.head()\n",
        "\n",
        "df0=df[:50]\n",
        "df1=df[50:100]\n",
        "df2=df[100:150]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EQvlykw0H5Cg",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "cellView": "form",
        "outputId": "b874560c-bdd6-436c-9783-8a73b154d484"
      },
      "source": [
        "#@title\n",
        "plt.xlabel('Sepal Length')\n",
        "plt.ylabel('Sepal Width')\n",
        "plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color=\"green\",marker='+')\n",
        "plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color=\"blue\",marker='.')"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f0399dd3128>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2gv_UTkPH9OU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "5a58c1a7-83ee-45a0-9d12-9672ddb55ad6"
      },
      "source": [
        "plt.xlabel('petal Length')\n",
        "plt.ylabel('petal Width')\n",
        "plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color=\"green\",marker='+')\n",
        "plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color=\"blue\",marker='.')"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7f039754e9b0>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\nl48crDcAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "39P9ABT6IOAU",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x=df.drop(['target','flower_name'],axis=1)\n",
        "y=df.target\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jaadFi4zJgEB",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "f5577613-dd43-46ed-dc7f-057b7b73d010"
      },
      "source": [
        "len(x_train)"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "120"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9Ykbw1WMJlhB",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "f233a01e-c452-4405-c299-17d0ee810b39"
      },
      "source": [
        "len(x_test)"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "30"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "i6YSClj_Jpq3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.svm import SVC\n",
        "model=SVC()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3-55CcusJ27Z",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 88
        },
        "outputId": "1724730b-d36b-46d8-a3b8-2f4890f3808b"
      },
      "source": [
        "model.fit(x_train, y_train)\n",
        "model.score(x_test,y_test)"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.\n",
            "  \"avoid this warning.\", FutureWarning)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "00BrhPM4KtoX",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "cellView": "form",
        "outputId": "e8059e2e-4cb0-4ea0-971e-c579ca6fa85c"
      },
      "source": [
        "#@title\n",
        "model.predict([[4.8,3.0,1.5,0.3]])"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    }
  ]
}