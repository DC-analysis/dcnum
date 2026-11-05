.. dcnum documentation master file, created by
   sphinx-quickstart on Fri Jun  2 23:46:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dcnum's documentation!
=================================

dcnum is a numerics toolbox for imaging deformability cytometry (DC). It
processes raw DC measurement data (.rtdc files): it computes background
images, segments the events in each frame (by thresholding or with
machine-learning models), extracts features such as area, deformation,
brightness and texture, and writes the results to an .rtdc file that can
be opened with e.g. `dclab <https://dclab.readthedocs.io>`_ or
`Shape-Out <https://shapeout.readthedocs.io>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   sec_getting_started
   sec_pipeline
   sec_design
   sec_changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
