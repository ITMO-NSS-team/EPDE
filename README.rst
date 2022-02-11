We use this repository as the main entry point for the EDPE framework.

We are of opinion that partial differential equations (PDE) discovery is more that the regression in the prescribed differential terms space. We propose the novel method of evolutionary equation discovery. Apart from the differential equations we have the algrbraic expressions discovery algorithm that is now uses same abstractions.


The project is maintained by the research team of the Natural Systems Simulation Lab, which is a part of the National Center for Cognitive Research of ITMO University.

The intro video about EPDE is available here:


.. image:: https://res.cloudinary.com/richarddedekind/image/upload/v1623953761/EDPE_front_dsyl9h.png
   :target: https://www.youtube.com/watch?v=BSXGCeuTcdc
   :alt: Introducing EPDE

EPDE features
==============

The main features of the framework are as follows:

- We dont need to create prescribed library of terms - our 'building blocks' are single differential terms or simple functions
- We have the multi-objective version that allows to obtain Pareto frontier of the equations, which helps the expert to choose one from several equation. 
- We use our own numerical differntiaion scheme, which allows to deal with high noise values
- We have solver to visualize the differential equations discovery process to add more control (https://github.com/ITMO-NSS-team/torch_DE_solver)

Installation
============

Common installation:

.. code-block::

  $ pip install epde


Examples & Tutorials
====================



Citations
============


@article{maslyaev2021partial,
  title={Partial differential equations discovery with EPDE framework: application for real and synthetic data},
  author={Maslyaev, Mikhail and Hvatov, Alexander and Kalyuzhnaya, Anna V},
  journal={Journal of Computational Science},
  pages={101345},
  year={2021},
  publisher={Elsevier}
}


@article{maslyaev2019discovery,
  title={Discovery of the data-driven differential equation-based models of continuous metocean process},
  author={Maslyaev, Mikhail and Hvatov, Alexander},
  journal={Procedia Computer Science},
  volume={156},
  pages={367--376},
  year={2019},
  publisher={Elsevier}
}


