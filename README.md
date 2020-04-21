# [ICCAD2020 Problem C](http://iccad-contest.org/2020/problems.html)
## GPU Accelerated Logic Re-simulation

## Usage

1. Preprocess:  
`python GraphPreprocessing.py <gv_file> <sdf_file> <sdf_lib> <intermediate_file>`

2. Build Simulator  
`./script/build`

3. Run Simulation:  
`./build/GPUSimulator <intermediate_file> <input_vcd> <SAIF or VCD Flag> <output_file>` 