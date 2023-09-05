# Berlin Cyclist Accidents Analysis - mid-term project

## Description

This project aims to analyze the relationship between population density, geographical location, and the number of cyclists involved in road accidents in Berlin.

## Hypotheses

1. In districts with higher population density, the number of cyclists involved in accidents on the roads is higher.
2. In districts that are near the city center, the number of cyclists involved in accidents on the roads is higher.

## Datasets Used

- Berlin road accident data - https://daten.berlin.de/datensaetze
    - Amt für Statistik Berlin Brandenburg / [Straßenverkehrsunfälle nach Unfallort in Berlin 2018]
    - Amt für Statistik Berlin Brandenburg / [Straßenverkehrsunfälle nach Unfallort in Berlin 2019]
    - Amt für Statistik Berlin Brandenburg / [Straßenverkehrsunfälle nach Unfallort in Berlin 2020]
    - Amt für Statistik Berlin Brandenburg / [Straßenverkehrsunfälle nach Unfallort in Berlin 2021]

- Berlin population data - https://www.statistik-berlin-brandenburg.de/a-i-16-hj
    - Amt für Statistik Berlin Brandenburg / [A | 16 - hj 2/22]

- Berlin geographical data - https://daten.odis-berlin.de/ 
    - "Geoportal Berlin / [lor_bezirksregionen_2021]
    - "Geoportal Berlin / [lor_ortsteile]
    - "Geoportal Berlin / [lor_planungsraeume_2021]
    - "Geoportal Berlin / [lor_prognoseraume_2021]
    - "Geoportal Berlin / [Detailnetz-Strassenabschnitte] - for further analysis

- Berlin geographical data - https://fbinter.stadt-berlin.de/ - didn't use in the final approach 
    - Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen / [Flächennutzung, Stadtstruktur 2020 und Versiegelung 2021 (Umweltatlas)] 

- Berlin Adress library - https://www.statistik-berlin-brandenburg.de/
    - Amt für Statistik Berlin Brandenburg / [Adressverzeichnis für die lebensweltlich orientierten Räume Berlin] - for all districts

### Final Documents

1. `main.ipynb`  
    - Main notebook that consolidates all work and analyses.

2. `project_functions.py`  
    - Python script containing utility functions and libraries used in the project.

### Working Documents (For Educational Purposes)

1. `0_adress_data_import.ipynb`  
    - Importing and initial investigation of address data.

2. `0_import_accident_data.ipynb`  
    - Importing and initial investigation of Berlin road accident data.

3. `0_import_geo_data.ipynb`  
    - Importing and initial investigation of geographical data.

4. `0_import_population_dataset.ipynb`  
    - Importing and initial investigation of population data.

5. `1_data_wrangling_and_eda.ipynb`  
    - Data wrangling and exploratory data analysis.

6. `2_model_to_predict_empty_LOR_Values.ipynb`  
    - Model to predict missing LOR (sub districts) values.

7. `3_hypothesis_testing.ipynb`  
    - Notebook focusing on the hypothesis testing.

### Unsuccessful Initial Approach

1. `X_adress_data_import.ipynb`  
    - Initial approach to importing and investigating address data.

2. `X_adress_data_manipulation.ipynb`  
    - Initial approach to manipulating address data.

3. `X_urbun_structure_import.ipynb`  
    - Initial approach to importing and investigating urban structure data.


## How to Run the Project

1. Clone the repository or download the project files.
2. Install the required packages.
3. Run `main.ipynb`

## Data Summary - final presentation 

https://public.tableau.com/app/profile/domenic4547/viz/berlin_cycling_accidents/Story1

## License
This project is licensed under the Creative Commons Attribution 3.0 Germany (CC BY 3.0 DE) License. For more details, please visit [Creative Commons CC BY 3.0 DE License](https://creativecommons.org/licenses/by/3.0/de/).



