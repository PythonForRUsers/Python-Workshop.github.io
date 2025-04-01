# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "|3,4|6"
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def speak(self):
        return f"{self.name} says woof!"
#
#
#
#
#
#
#
#
#| code-line-numbers: "1|2|"
#| output-location: fragment
my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.speak())  
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
class Dog: ## class definition
    def __init__(self, name, breed): ## sets up the initialization for an instance of class Dog. 
        ### Allows us to assign name and breed when we instantiate dog. 
        self.name = name ## attributes
        self.breed = breed

    def speak(self): ## method
        return f"{self.name} says Woof!"


#
#
#
#
#
#
#
#| output-location: fragment
my_dog = Dog("Fido", "Labrador") ## create a dog of name 'Fido' and breed 'Labrador'
print(my_dog.speak())

## if we want to see what kind of dog our dog is
print(f"Our dog {my_dog.name} is a {my_dog.breed}.")
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "|1|2-7|"
class GuardDog(Dog):  # GuardDog inherits from Dog
    def __init__(self, name, breed, training_level): ## in addition to name and breed, we can 
        # define a training level. 
        # Call the parent (Dog) class's __init__ method
        super().__init__(name, breed)
        self.training_level = training_level  # New attribute for GuardDog that stores the 
        # training level for the dog

    def guard(self): ## checks if the training level is > 5 and if not says train more
        if self.training_level > 5:
            return f"{self.name} is guarding the house!"
        else:
            return f"{self.name} needs more training before guarding."
    
    def train(self): ## modifies the training_level attribute to increase the dog's training level
        self.training_level = self.training_level + 1
        return f"Training {self.name}. {self.name}'s training level is now {self.training_level}"

# Creating an instance of GuardDog
my_guard_dog = GuardDog("Rex", "German Shepherd", training_level= 5)
```
#
#
#
#
#| slide-type: fragment
#| output-location: fragment
# Using methods from the base class
print(my_guard_dog.speak())  # Inherited from Dog -> Output: "Rex says Woof!"
```
#
#| slide-type: fragment
#| output-location: fragment
# Using a method from the derived class
print(f"{my_guard_dog.name}'s training level is {my_guard_dog.training_level}.")
print(my_guard_dog.guard()) 
```
#
#| slide-type: fragment
#| output-location: fragment
## if we want to train Rex and increase his training level, 
print(my_guard_dog.train())

## now check if he can guard 
print(my_guard_dog.guard()) 
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "|2,3|4|6-17|"
class TrickMixin: ## mixin that will let us teach a dog tricks
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Ensures proper initialization in multiple inheritance
        self.tricks = []  # Store learned tricks

    def learn_trick(self, trick):
        """Teaches the dog a new trick."""
        if trick not in self.tricks:
            self.tricks.append(trick)
            return f"{self.name} learned a new trick: {trick}!"
        return f"{self.name} already knows {trick}!"

    def perform_tricks(self):
        """Returns a list of tricks the dog knows."""
        if self.tricks:
            return f"{self.name} can perform: {', '.join(self.tricks)}."
        return f"{self.name} hasn't learned any tricks yet."

## note: the TrickMixin class is not a standalone class! it does not let us create a dog on its own!!!
```
#
#
#
#
#
#
#
#
#
#| slide-type: fragment
#| output-location: fragment
class SmartDog(Dog, TrickMixin):
    def __init__(self, name, breed):
        super().__init__(name, breed)  # Initialize Dog class
        TrickMixin.__init__(self)  # Initialize TrickMixin separately
#
#
#
#
#
#| slide-type: fragment
#| output-location: fragment
# a SmartDog object can use methods from both parent object `Dog` and mixin `TrickMixin`.
my_smart_dog = SmartDog("Buddy", "Border Collie")
#
#
#
#
#
#| slide-type: fragment
#| output-location: fragment
print(my_smart_dog.speak()) 
#
#
#
#
#
#| slide-type: fragment
#| output-location: fragment
print(my_smart_dog.learn_trick("Sit"))  
print(my_smart_dog.learn_trick("Roll Over")) 
print(my_smart_dog.learn_trick("Sit"))  

```
#
#
#
#| slide-type: fragment
#| output-location: fragment
print(my_smart_dog.perform_tricks()) 
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "|5-6, 12-13"
class Human:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} says hello!"

class Parrot:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} says squawk!"

#
#
#
#
#
#
#
#| output-location: fragment
def call_speaker(obj):
    print(obj.speak())

call_speaker(Dog("Fido", "Labrador"))
call_speaker(Human("Alice"))
call_speaker(Parrot("Polly"))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "|12|14-17|19-21|"
## imports
import pandas as pd
import numpy as np

from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt

from great_tables import GT
from tabulate import tabulate

## sklearn imports

## import classes
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

## import functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
#
#
#
#
#
#| code-line-numbers: "1,2|4-11|13-16|18-19|"
#| output-location: slide
# Load the Penguins dataset
penguins = sns.load_dataset("penguins").dropna()

# Make a summary table for the penguins dataset, grouping by species. 
summary_table = penguins.groupby("species").agg({
    "bill_length_mm": ["mean", "std", "min", "max"],
    "bill_depth_mm": ["mean", "std", "min", "max"],
    "flipper_length_mm": ["mean", "std", "min", "max"],
    "body_mass_g": ["mean", "std", "min", "max"],
    "sex": lambda x: x.value_counts().to_dict()  # Count of males and females
})

# Round numeric values to 1 decimal place (excluding the 'sex' column)
for col in summary_table.columns:
    if summary_table[col].dtype in [float, int]:
        summary_table[col] = summary_table[col].round(1)

# Display the result
display(summary_table)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-line-numbers: "1-3|5-7|"
# Selecting features for clustering -> let's just use bill length and bill depth.
X = penguins[["bill_length_mm", "bill_depth_mm"]]
y = penguins["species"]

# Standardizing the features for better clustering performance
scaler = StandardScaler() ## create instance of StandardScaler
X_scaled = scaler.fit_transform(X) ## same as calling scaler.fit(X) then X_scaled = scaler.transform(X)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| output-location: fragment
## Choosing 3 clusters b/c we have 3 species
kmeans = KMeans(n_clusters=3, random_state=42) ## make an instance of the K means class
#
#
#
#
#
#| code-line-numbers: "1-2|4-5|"
#| output-location: fragment
## the fit
penguins["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

## now that we fit the model, we should have cluster centers
print("Coordinates of cluster centers:", kmeans.cluster_centers_)
#
#
#
#
#
#
#
#
#
#| output-location: fragment
# Plotnine scatterplot of species by bill length/depth
plot1 = (ggplot(penguins, aes(x="bill_length_mm", y="bill_depth_mm", color="species"))
 + geom_point()
 + ggtitle("Penguin Species")
 + theme_bw())

display(plot1)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| output-location: fragment
# Plotnine scatterplot of k-Means clusters
plot2 = (ggplot(penguins, aes(x="bill_length_mm", y="bill_depth_mm", color="factor(kmeans_cluster)"))
 + geom_point()
 + ggtitle("K-Means Clustering Results")
 + theme_bw())

display(plot2)
```
#
#
#
#
#
#
#
#
#
#| output-location: fragment
# Calculate clustering performance using Adjusted Rand Index (ARI)
kmeans_ari = adjusted_rand_score(penguins['species'], penguins["kmeans_cluster"])
print(f"k-Means Adjusted Rand Index: {kmeans_ari:.2f}")
#
#
#
#
#
#
#
#| code-line-numbers: "1-3|5-20|"
#| output-location: slide
# Count occurrences of each species-cluster-sex combination
# ( .size gives the count as index, use reset_index to get count column. )
scatter_data = penguins.groupby(["species", "kmeans_cluster", "sex"]).size().reset_index(name="count")

# Create a heatmap of the cluster assignments by species
heatmap_plot = (
    ggplot(scatter_data, aes(x="species", y="kmeans_cluster", fill="count"))
    + geom_tile(color="white")  # Add white grid lines for separation
    + scale_fill_gradient(low="lightblue", high="darkblue")  # Heatmap colors
    + labs(
        title="Heatmap of KMeans Clustering by Species",
        x="Species",
        y="KMeans Cluster",
        fill="Count"
    )
    + theme_bw()
)

# Display the plot
display(heatmap_plot)
#
#
#
#
#
#
#
#| output-location: fragment
scatter_plot = (
    ggplot(scatter_data, aes(x="species", y="kmeans_cluster", color="species", shape="sex", size="count"))
    + geom_point(alpha=0.7, position=position_dodge(width=0.5))  # Horizontal separation
    + scale_size(range=(2, 10))  # Adjust point sizes
    + scale_y_continuous(breaks=[0, 1, 2])  # Set y-axis ticks to only 0, 1, 2
    + theme_bw()
    + labs(
        title="KMeans Clustering vs Species (Size = Count)",
        x="Species",
        y="KMeans Cluster"
    )
)
#Display the plot
display(scatter_plot)
#
#
#
#
#
#
#
#
#
#
#
# Splitting dataset into training and testing sets (still using scaled X!)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| output: false
## perform knn classification
# Applying k-NN classification with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5) ## make an instance of the KNeighborsClassifier class
# and set the n_neighbors parameter to be 5. 

# Use the fit method to fit the model to the training data
knn.fit(X_train, y_train)
#
#
#
#
#
#
#
#| output-location: fragment
print(knn.classes_)
#
#
#
#
#
#| output-location: fragment
# Use the predict method on the test data to get the predictions for the test data
y_pred = knn.predict(X_test)

# Also can take a look at the prediction probabilities, 
# and use the .classes_ attribute to put the column labels in the right order
probs = pd.DataFrame(
    knn.predict_proba(X_test),
    columns = knn.classes_)
probs['y_pred'] = y_pred

print("Predicted probabilities: \n", probs.head())
#
#
#
#
#
#
#
#
## First unscale the test data
X_test_unscaled = scaler.inverse_transform(X_test)

## create dataframe 
penguins_test = pd.DataFrame(
    X_test_unscaled,
    columns=['bill_length_mm', 'bill_depth_mm']
)

## add actual and predicted species 
penguins_test['y_actual'] = y_test.values
penguins_test['y_pred'] = y_pred
penguins_test['correct'] = penguins_test['y_actual'] == penguins_test['y_pred']
#
#
#
#
#
#| output-location: fragment
## Build the plot
plot3 = (ggplot(penguins_test, aes(x="bill_length_mm", y="bill_depth_mm", 
color="y_actual", fill = 'y_pred', size = 'correct'))
 + geom_point()
 + scale_size_manual(values={True: 2, False: 5})
 + ggtitle("k-NN Classification Results")
 + theme_bw())

display(plot3)
#
#
#
#
#
#| output-location: fragment
## eval knn performance
# Calculate accuracy and print classification report -> 
# accuracy_score and classification_report are functions! 
knn_accuracy = accuracy_score(y_test, y_pred)
print(f"k-NN Accuracy: {knn_accuracy:.2f}")
print(classification_report(y_test, y_pred))
#
#
#
#
#
#| output-location: fragment
##  making a summary table
# Creating a summary table
summary_table = pd.DataFrame({
    "Metric": ["k-Means Adjusted Rand Index", "k-NN Accuracy"],
    "Value": [kmeans_ari, knn_accuracy]
})
GT(summary_table).show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
