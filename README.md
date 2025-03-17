# E-commerce-Return-task-for-data-mining


KAIST AI 506 Data mining course final project  <br>

I worked on a project in a data mining class to predict return status based on order information. Using 850,000 order records, we framed this as a multi-class classification problem with three categories: full return, partial return, and no return. The tabular dataset included features like product type, weight, color, and customer details.

I approached this problem using a graph-based method, focusing on how to define edges between orders. To construct the graph, I proposed a similarity-based approach, where customers were connected based on the characteristics of the products they purchased. Specifically, I created customer feature vectors from product attributes and computed Jaccard Similarity to form edges. Finally, I implemented a node classification model using this graph and trained it for prediction.
**This code cannot be executed because it does not contain data.**

![Method](https://github.com/user-attachments/assets/27df50ba-a456-4247-956f-3a23be69bb23)


<br>

![image](https://github.com/user-attachments/assets/dbbe0d53-9ada-4c8b-9b4c-def8f67c42ad)
