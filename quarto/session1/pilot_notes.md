# Notes from session 1

Session went well! We can streamline it though. 

## Some notes: 

1. we should probably skip talking about R and stuff and get right to the tutorial. I think the more we talk before we actually install things the more questions there will be. 
2. Might be worth downloading everything first and then talking through the intro. It can take a while for things to download!

We need to keep everyone on track and not get distracted ourselves. (Use sticky note method: everyone puts sticky note when they are done with step, move on when ~90% of people have doen step.)

## On the question of "Will this break my R?"
No it won't. R studio pretty much isolates itself. You can check what environment R studio is running on with the R Studio Terminal (and just use conda deactivate to get out of an environment if it is activated).


# Updates 2025

1. Utilize the previous Session 1: Python installation tutorial as a pre-session handout:
    * Just like any package documentation, the very first thing is code/links for installating software
    * Give basic overview of Python / Jupyter Notebook (should we discard Quarto?)/ IDE (i.e.,VS code)
    * Introduce the concept of **virtual environment**--for the purpose of keeping a reproducibile workflow (just like working in R)

2. Session 1: Goal is to set up a project folder from which people do their leanring practices for the workshop
    * Recap of main concepts: Python (language: the foundation) characteristics, VS code (the code editor where you *code*), **and** Anaconda/Conda (your virtual environment manager)
    * Demonstrate steps of installing miniconda virtual env locally with command line (by then we will have conda installed)
    * Setup Workflow: 
        * Github create repo -> clone repo to local (H/C drive) -> open folder in VS code ->  activate conda env -> create files/folders

    ```
    ├── README.md
    ├── environment.yaml
    ├── data-date.txt
    ├── .gitignore
    ├── data/
    └── scr/
    ```