# Innovation Topology

This package contains code for measuring scientific papers which close homological holes in concept networks derived 
from the Microsoft Academic Graph (MAG) dataset.

We are interested in papers which bridge conceptual gaps in the scientific network of concepts as it grows through time. 
We create this scientific concept network by connecting two concepts with an edge if those concepts appear within the 
same paper.
The set of concepts is extracted by MAG, known in the datset as "Fields of Study". 
These Fields of Study are hierarchical in specificity, with Level 0 corresponding to high-level subjects ("Sociology", "Physics", "Economics", etc.) and Level 5 corresponding to highly specific concepts like "Turan's inequalities". 
The SQL script in `innovationtopology/mag/prepare_mag_networks.sql` will create a concept network connecting concepts within each hierarchical level. 
This script assumes the MAG data resides on a sql-like server containing the relevant tables of the MAG dataset. 
Because it creates networks across all conceptual levels across the entire MAG dataset, the script may take upwards of a day to 
run, depending on the underlying hardware of the server. 

Once these concept networks have been created, we can proceed with measuring the gaps in these concept networks and the papers which combine concepts which bridge these gaps. 
To do this, we apply persistent homology--calculated using the (Gudhi package)[https://gudhi.inria.fr]--to a temporal filtration of the concept network defined. 
This temporal filtration is defined by adding an edge between concepts based on the date of the earliest paper to combine the two concepts, assuming the paper has paper-concept Score greater than a threshold (set at 0.6 for the example data in this repo). 
By applying persistent homology to this temporal filtration, we can measure the higher-order conceptual gaps in the network of scientific concepts within each Level 0 subject along with the papers which close these conceptual gaps at particular points in history.
The script `innovationtopology/mag/hole_closing_papers.py` computes this temporal filtration, calculates persistent homology over this filtration, and records the hole-closing concepts and papers for each hierarchical level. 
An example output of this process is provided in `data/temporal_holes-fields_of_study/close_0.6_ep0.1_dim<dimension>_sociology_<level>.csv`. 

While the `hole_closing_papers.py` script calculates the persistence pairs which define the hole-closing concept pairs, we 
may also be interested in what the conceptual gap was which was closed by the paper which combined these concepts which define the persistence pair. 
Unfortunately, the Gudhi persistent homology package does not provide information on _representative cycles_--the set of concepts which define this conceptual gap. 
In order to compute these representative cycles, we use the `tight_representative_cycles` branch of (Ripser)[https://github.com/Ripser/ripser/tree/tight-representative-cycles] which provides something close to a minimal cycle representation of the concepts involved in this conceptual gap. 
The `innovationtopology/mag/ripser_tight_representatives.py` script computes these representative cycles and outputs a python pickle object containing these homological representatives. 
An example of this output may be found at `data/temporal_holes-fields_of_study/ripser_tight_representatives/0.6_ep0.1_dim<dimension>_sociology_<level>_1.pkl`
Note that this script depends on the output of the `hole_closing_papers.py` script to run. 

Finally, we can combine the outputs of the two scripts using the `innovationtopology/mag/map_representative_cycles.py`. 
This script will create a (networkx)[https://networkx.org/] representation of the representative cycles and will match the persistence pairs to their representative cycles. 
Note that this matching is subject to some ambiguity, as multiple holes may be closed at a particular date in the filtration, and that there is a choice of persistence pairs whch may differ between the Gudhi and Ripser packages. 
An example of such a file is available at `data/temporal_holes-fields_of_study/ripser_tight_representatives/joined/0.6_ep0.1_dim<dimension>_sociology_<level>_1.csv/pkl`

## To Run

This repository was tested with python3.10.
From the top level of the package directory, install the required dependencies with 

```bash
pip install -r requirements.txt
pip install -e .
```

As mentioned above, the `innovationtopology/mag/hole_closing_papers.py` script depends on the network representation 
constructed by the `innovationtopology/mag/prepare_mag_networks.sql`, so this script must be run prior to running the python scripts.

To access these networks, the python code uses pymysql as the SQL engine. 
The code looks for username and password information for the MySQL server in the `innovationtopology/config.py` file under 
the `DB_CONFIG` variable, so one must update this information before running any of the scripts which pull data from the SQL server.

The code also requires a pointer to an installation of the (`tight_representative_cycles` branch of Ripser)[https://github.com/Ripser/ripser/tree/tight-representative-cycles] 
which should be provided in the `config.py` file by setting the `RIPSER_LOC` variable to the `ripser_tight_representatives` executable. 

To run the python scripts, enter the `innovationtopology` subdirectory and run the desired script, e.g.
```python
python mag/hole_closing_papers.py
```
