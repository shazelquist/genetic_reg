# genetic_reg

An exploration of genetic algorithms in search regular expressions.

Project goals:
 - Generate regular expressions for searches
 - Examine the influence of non-expressed genes
 - Modify the genetic algorithm & measure effectiveness

This project is ported from an assignment, and generally for personal use.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Graphics
#### Every gene is expressed
![coding](https://user-images.githubusercontent.com/11480905/174935372-a290023e-7239-4f37-bc28-e8ab0593f3e8.png)
#### Sequences can contain non-expressed genes
![non_coding](https://user-images.githubusercontent.com/11480905/174935389-5e2d2b36-312d-4ff2-ae3a-35c256f88af3.png)
#### Direct comparison
![comparing_coding_noncoding](https://user-images.githubusercontent.com/11480905/174936520-963f40a1-7299-470c-908f-66a7ff435fa0.png)

## Run quick start
### Installation:
 1. ``$ git clone https://github.com/shazelquist/genetic_reg.git``
You may wish to edit the hashbangs in any python file to your configuration.
 2. ``pip install -r requirements.txt``
Using a virtualenv or some other package manager like conda is highly recomended.
You may also need to change file permissions ``$ chmod u+x *.py``

## Let's get started
``$ ./gen_reg.py sourcetext_filename, Target, Populations, Operation_sequence``
Details on valid parameters will be made avaliable.
You also may want to test with default parameters.
``$ ./gen_reg.py``
