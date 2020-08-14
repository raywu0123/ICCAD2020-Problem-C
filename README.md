# [ICCAD2020 Problem C](http://iccad-contest.org/2020/problems.html)
## GPU Accelerated Logic Re-simulation

## Usage

### Setup
* Initialize Git Submodules:
```
git submodule init && git submodule update
``` 
* python interpreter version for pyenv is specified in `.python-version`.
You may delete this file if you want to use your own interpreter version. 
* Use pipenv to setup python environment.

### Simulation
1. Preprocess:  
```
pipenv run python GraphPreprocessing.py <gv_file> <sdf_file> <std_lib> <intermediate_file>
```

2. Build simulator  
```
./scripts/build
```

3. Run simulation:  
```
./build/GPUSimulator <intermediate_file> <input_vcd> <dumpon_time> <dumpoff_time> [output_file] [SAIF or VCD Flag]
```

### Testing
* Run unit tests  
```
./scripts/test.sh
```

* Run end-to-end tests    
```
pipenv run python ./custom-tests/test-custom-cases.py [-s path_to_results]
```

### Export binaries
```
./scripts/export_binaries <output_dir>
```
