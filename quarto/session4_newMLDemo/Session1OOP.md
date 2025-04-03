# **1. Programming in Python – R vs Python**  

Both R and Python are high-level programming languages that allow us to perform a wide range of data analysis and computational tasks. There are many debates about which language is better for specific tasks, but in reality, both are powerful tools, and most things you can do in R, you can also do in Python, and vice versa.

Like R, Python has **data structures** and **functions**:  
- **Data structures** are ways of organizing and storing data so it can be accessed and modified efficiently. Examples in Python include lists, dictionaries, tuples, and sets, while R has vectors, lists, data frames, and matrices.  
- **Functions** are reusable blocks of code designed to perform a specific task. Python and R both allow users to define their own functions or use built-in functions.  

Both languages also support installing external **packages (libraries)** to extend functionality. In Python, this is typically done through `pip`, whereas R uses `install.packages()`.  

### **Why Learn Python?**  
If R and Python are so similar, why should we bother learning Python?  

While R is extremely powerful for statistical computing and data visualization, Python dominates in areas like:  
- **Machine learning** (scikit-learn, TensorFlow, PyTorch)  
- **Neural networks and deep learning**  (TensorFlow, PyTorch)
- **Natural language processing** (NLP)  
- **Image analysis**  
- **Web scraping** (BeautifulSoup, Scrapy)  
- **Genomic and transcriptomic analysis** (Biopython, Scanpy)  

Many cutting-edge tools in bioinformatics, AI, and automation are written in Python. By learning Python, we can expand our capabilities beyond statistical analysis and gain access to a broader ecosystem of tools.  

The goal of these workshops is to build familiarity with Python and provide the skills needed to integrate it into your work.  

---

# **2. Differences Between R and Python**  

### **Programming Paradigm: Functional vs Object-Oriented**  
One of the biggest differences between R and Python is **how they structure code**:  

- **R is function-oriented**: R code is typically structured around functions that take input, process it, and return output. Even though R does have object-oriented programming (OOP) features, they are not as commonly used.  
- **Python is more object-oriented**: Python structures code around **objects**. This means that Python uses **classes** as blueprints to create objects, and objects contain both **attributes** (data) and **methods** (functions that act on objects).  

In R, you can use vectors, data frames, and functions without thinking much about objects and classes. In Python, however, **everything is an object!** Even basic data structures (lists, dictionaries, etc.) are instances of classes.  

For example, in Python, a `LogisticRegression` from `scikit-learn` is a class. When we create a logistic regression model, we are making an **instance** of that class, which has specific properties and methods.  

### **Library Management & Function Namespacing**  
- **R** allows functions from different packages to be used without specifying which package they come from (`library(dplyr)`).  
- **Python** enforces stricter namespacing: functions must be called using the package they belong to (`pandas.DataFrame()` instead of just `DataFrame()`).  

Example:
```python
import re  # Importing the regular expressions module
re.sub(pattern, replacement, text)  # Function call
```
This is similar to calling `stringr::str_replace()` in R instead of just `str_replace()`.

While it may seem cumbersome, **Python’s strict namespacing helps prevent function conflicts** between different libraries.  

---

# **3. Quick Primer on Object-Oriented Programming (OOP)**  

Object-Oriented Programming (OOP) organizes code by bundling **data (attributes)** and **behaviors (methods)** into objects.  

For example, imagine an object representing a **person**:
- **Attributes**: Name, age, address  
- **Methods**: Walk, talk, eat  

Or a **logistic regression model**:
- **Attributes**: Coefficients, intercept  
- **Methods**: `.fit()`, `.predict()`, `.score()`  

### **Key Principles of OOP**  

1. **Encapsulation**: Bundling data and methods together in a single unit.  
   - Example: A `DataFrame` object in pandas has its own methods for manipulation (`.sort_values()`, `.groupby()`).  
   - Benefit: Prevents unintended modifications and keeps code organized.  

2. **Inheritance**: Creating new classes based on existing ones.  
   - Example: `sklearn.LinearRegression` inherits from a general regression model class.  
   - Benefit: Code reuse and modular design.  

3. **Abstraction**: Hiding implementation details and exposing only essential functionality.  
   - Example: `.fit()` is implemented differently for logistic regression and KNN, but the user interacts with it the same way.  

4. **Polymorphism**: Objects of different types can be treated the same way if they implement the same methods.  
   - Example: Python’s **duck typing**:  
     - *"If it walks like a duck and quacks like a duck, then it must be a duck."*  
     - If an object has the right attributes and methods, it can be used in the same way as another object of a different class.  

### **How This Relates to Python Data Structures** 
**We will cover this next session!** But for now:  
- Lists, dictionaries, data frames—everything in Python is an object!  
- When you create a list (`my_list = []`), you’re actually making an instance of Python’s built-in `list` class.  
- Data structures in Python have **methods** that allow you to interact with them (`.append()`, `.sort()`, etc.).  

---
### **Final Thoughts**  
Python and R are both powerful languages, and learning Python **doesn’t mean you should stop using R**. Instead, adding Python to your skill set expands your capabilities, especially in machine learning, automation, and working with large datasets.  

--------------- IGNORE BELOW !!!------------------

# 1. Programming in Python - R vs Python

Both R and python are high level programming languages that allow us to do many things, and there are many debates over which one is better for any given task. Like R, python has data structures (defn) and functions (defn). We can also install import packages (libraries) and call functons associated with them. In fact, many if not most, of the things we can do in R we can do in Python and vice versa. 

If that is the case, why do we want to learn python? 

While R is an extremely powerful language for data and statistical analysis, much cutting edge work in machine learning, image analysis, neural networks and natural language processing is done in Python. Python is also great for web scraping and has many tools for working with text data. Many genomic and transcriptomic tools are also built in python.

The goal for these workshops is to build familiarity with python and give the tools needed for using python in your work. 

# 2. Differences between R and Python

One of the main differences between python and R is that python is more object oriented (although R is also technically object oriented). This means that, instead of using functions as the main unit of code, python uses classes. Classes are blueprints for creating objects, where objects are instances of classes that contain specific data. As part of the 'blueprint', classes contain methods (functions that can act on objects of that class) and attributes (variables belonging to objects of that class). While classes and objects exist in R (every vector you make is an instance of the class `vector`!), it is possible to do a lot of programming in R without ever having to worry about them. This is not the case in python!

Similar to R, Python allows us to install libraries containing useful functions that are not available in the base installation. Unlike R, however, python functions are installed via the command line! Python is also more careful than R about keeping functions attached to the libraries they come from, and functions that are designed to act on specific objects (like a dataframe) tend to be written as methods (functions definined in the class blueprint) rather than standalone functions. For example, when you want to use the string substitution function `sub` from the `re` library in python, you call the function as `re.sub()` rather than just `sub`. This is similar to calling `stringr::sub()` in R. This might seem annoying, but it serves to help avoid conflicts between libraries. 

Like functions, classes can also come as part of libraries in python. For example, the scikit-learn package gives us the `LogisticRegression` class, which serves as a blueprint for creating logistic regression models. In this example, a specific model would be an *instance* of the logistic regression class, and it would have *attributes* (variables) and *methods* (functions) associated with it that were defined in the class blueprint. 


# 3. Quick Primer on Object-Oriented Programming

Object-oriented programming (OOP) structures programs so that properties and behaviors are attached to individual objects rather than abstract concepts. For example, an object could represent a specific person, with properties such as a name, date of birth and address. This person object could also have behaviors like walking, talking and running. Objects can also be things like regression models with properties like coefficients and formulas and behaviors like fitting and predicting.

The OOP concept exists in other programming languages and is usually described as having 4 main tenants:
1. Encapsulation: bundling data (attributes) and behaviors (methods) into a cohesive unit. 
    - Helps control access to attributes
    - Makes it **clear** what behaviors can be performed on/by a specific object. (It's easy to try to apply a function to the wrong object type and have it fail in R. In python, most type-specific functions are written as methods which can be expected to work on objects of any class containing or inheriting those methods.)
2. Inheritance: hierarchical relationships between classes.
    - child classes inherit attributes and methods from parent classes
    - building off of a parent class promotes code reuse and reduces duplication
3. Abstraction: hides implementation details and exposes essential functionality. 
    - allows for a consistant interface for interactions with objects
    - ex: the `.fit()` method might be implemented differently for a logistic regression model and a knn classifier, but the user can expect that this method will perform the desired functionality regardless of how it is implemented in the background.  
4. Polymorphism: treat objects of different types as instances of the same base type as long as they have the right interface and behaviors. 
    - python's `duck typing`: 'If it walks like a duck and it quacks like a duck, then it must be a duck.'
    - if an object contains the right attributes and methods for a model, then we can assume it is and will behave like a model. 

In python, all of the data structures (which we will cover in the next session) are defined as classes! Therefore when we create a data structure (like a list), we are making an instance of that class. 

pre-read: https://realpython.com/python3-object-oriented-programming/ 

