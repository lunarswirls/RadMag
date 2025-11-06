# Radar and Magnetic (RadMag) Sounding Theory

## Description
These are useful scripts for radar and magnetic sounding modeling and analysis.


## Installation

These tools can be installed as a Python package called "radmagpy" for easy access to functions and visualization tools. This was done so that one could organize files into separate directories and still be able to reference each file by defining a path relative to the package root name.

To install as a local branch (highly recommended. NOTE the "-e" flag to install as a local branch! Again, highly recommended you do not ignore the "-e" flag!)

```
pip install -e <path/to/your/local/python/package/directory containing setup.py>
```
NOTE: This is not a path to setup.py! This is the path to the root directory containing setup.py!

As an example, if you are user tycho and you want to use and edit your copy of the radmagpy package that you put in `/homes/tycho/code/radmagpy/src`, you would do:
```
pip install -e /homes/tycho/code/radmagpy/
```


## Usage
### No roughness model
To run with no roughness model
```
python full_test.py --roughness-model 'none' --outdir 'results'
```

### Simple roughness model
To run with a simple roughness-driven depolarization of `k ~ 0.5·slope²`
```
python full_test.py --roughness-model 'simple' --outdir 'results'
```

### Facet-only roughness model
To run with a facet-only model, using the RMS slopes only (no extra transfer) 
```
python full_test.py --roughness-model 'facet' --outdir 'results'
```

### Simplified Integral Equation Model (IEM)
To run with a frequency-dependent single-parameter depolarizer using
`F = exp[-(4πσ_h cosθ / λ)^2]`

A fraction `(1 - F)` of **OC** power is transferred into **SC**, plus an optional **dihedral** term that enhances SC:

- `OC_new = F * OC`
- `SC_new = SC + (1 - F) * OC + k_d * sin^2(θ) * OC_new`

Where:

| Symbol | Meaning |
|--------|---------|
| `σ_h`  | RMS height of surface roughness (meters) |
| `θ`    | Local incidence angle for each facet |
| `λ`    | Radar wavelength |
| `F`    | Smooth-surface coherent return factor |
| `k_d`  | Optional dihedral boost coefficient |
```
python full_test.py --roughness-model 'iem-lite' --outdir 'results'
```


## Visuals
TBD


## Authors and acknowledgment
Primary Author: [Dany Waller](danywaller.github.io)
- Email: [dany.c.waller@gmail.com](mailto:dany.c.waller@gmail.com)
