#! /bin/bash
rm build -rf
rm *.so
python setup.py build_ext --inplace
