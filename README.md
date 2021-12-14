# CSC2222
Artifact for CSC2222 UofT course


```
# Note the .cnt file was too large for GitHub and so you must unzip it first
# will inflate `EEGparticipant 015.cnt`
unzip EEGparticipant\ 015.zip

# This artifact requires Python 3.9 
python3.9 venv env
source env/bin/activate
pip install -r requirements.txt

# Help
python BSA_encoder.py -h
python Izhikevich_STDP.py -h

# Usage
# Arguments of interest are `bsa_fl`, `bsa_t`, `bsa_c` (others may be left as default)
# This will read in the .cnt file, split into epochs, spike encode and save as a .npy file
python BSA_encoder.py

# Arguments of interest are `izhi_num`, `brian_cpp`, (others may be left as default)
# This will read in the outputted .npy file from above and train a Izhikevich neuronal model with STDP synapse.
python Izhikevich_STDP.py 
```

