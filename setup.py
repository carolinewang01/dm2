#!/usr/bin/env python

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
   name='pymarl',
   version="1.0.0",
   author="oxwhirl",
   description="pkg cloned from oxwhirl/pymarl",
   packages=find_packages()
)
