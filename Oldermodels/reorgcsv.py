
import pandas as pandasForSortingCSV
import csv
import pandas as pd
  
# assign dataset
csvData = pandasForSortingCSV.read_csv("./reorganizedtrain.csv")
                                         
# displaying unsorted data frame
print("\nBefore sorting:")
print(csvData)
  
# sort data frame
csvData.sort_values(["label"], 
                    axis=0,
                    inplace=True)
  
csvData.to_csv("./newreorg.csv")

# displaying sorted data frame
print("\nAfter sorting:")
print(csvData)

