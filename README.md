# AIBE_INS_LAB_REP_CLASS

## What is the plan?

...

## Getting started

1. Clone repo into Folder: Folder Structure should be:

Base folder:
 - Repo base folder [CLONE WITHIN BASE FOLDER!]
 - data base folder called: aibe_ins_lab_recordings

2. Get your Environment started using the environment.yml using Anaconda for example
    Not sure if it works...

3. Create the data base folder with the specific name: aibe_ins_lab_recordings 
and drop the .zip file inside!

Base folder:
 - Repo base folder
 - output
 - aibe_ins_lab_recordings                  ***
  - zipfilename.zip

4. Launch Dataset creation File
go to classification -> dataset.py -> run:
    Option 1: Unzip and split raw data (RUN ONCE WHEN YOU HAVE JUST CLONED THE REPO!!!!!!)
    # get_data_split(path_to_recordings_base)

    here path_to_recordings_base is the path to your aibe_ins_lab_recordings folder ***


5. Also make an output folder called "output" on the same level as the aibe_ins_lab_recordings folder or the repo base folder for outouts of dataframes and plots


not finished yet, just raw and structuring, will need training / testing split (maybe not physically but json or so for indexing)


# Bad Channels:
Channel 9 for Leon Grasp... but not for Mohammad peace for example!?
-> check recordings for bad channels!

Checked... just remove Channel 9 from leons data. Franzi and Mohammads is are fine

Figure out how to deal with bad Channel...