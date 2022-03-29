# Quantum Digital Cooling
Simulate quantum digital cooling protocols on model systems.

Quantum Digital Cooling (QDC) is a class of digital qauntum computing methods for approximate ground state or thermal state preparation, introduced in Phys. Rev. A 104, 0124148 (2021) and described in Chapter 5 of *Strategies for braiding and ground state preparation in digital quantum hardware*  (ISBN 978-90-8593-521-6).

This repository incudes the code used for simulations presented in that paper.



## QDC simulations

### Scripts
Scripts to run simulations of standard QDC protocols are incuded in `./scripts/`.
The usage is documented in each script.
All scripts require `qdclib` to be installed (see above).
Scripts save results as a dictionary in a `.json` file, named using the parameters passed to the script, in the specified data folder.

### Data
A set of pre-computed simulation outputs is included in this repository, under `./data`.
The subdirectory structure of data follows the hierarchy:
```
`./data`
└── `TFIM` - simulated system model
    └── `logsweep` - protocol
        └── `DM` - simulator type. eventually more than 1 level (e.g. `continuous/DM/`)
            ├── `cooling` - initial state or other protocol details
            │    └── `K2JvB2L2.json` - data files
            ├── `reheating`
            └── `iterative`
```
