
## Get Data:
**At the root directory, run the following command line arguments:**
`python -m ID_generation.preprocessing.get_data`

## Data Preprocessing:
**At the root directory, run the following command line arguments:**
### Example use cases:
* To process Sports_and_Outdoors dataset without extracting attributes:
`python -m ID_generation.preprocessing.data_process -n Sports_and_Outdoors`

* To process Beauty, Sports_and_Outdoors, and Toys_and_Games with attribute extraction:
`python -m ID_generation.preprocessing.data_process -n Beauty Sports_and_Outdoors Toys_and_Games -a`

* To see a full list of options, use the help command:
`python -m ID_generation.preprocessing.data_process --help`
