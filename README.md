# Grocery Store Product Classification

This repository contains the implementation of a neural network that classifies smartphone pictures of grocery store products into one of 43 predefined categories. This project is divided into two parts:

1. **Implementing a custom neural network from scratch.**
2. **Fine-tuning a pretrained ResNet-18 model using PyTorch.**

## Project Overview

### Dataset
The dataset used for this project contains natural images of products taken with a smartphone camera in grocery stores. It is divided into three splits:
- **Train**
- **Validation (Val)**
- **Test**

The dataset includes images belonging to the following 43 product categories:

```
0.  Apple            1.  Avocado           2.  Banana          3.  Kiwi
4.  Lemon            5.  Lime              6.  Mango           7.  Melon
8.  Nectarine        9.  Orange           10. Papaya         11. Passion-Fruit
12. Peach           13. Pear             14. Pineapple      15. Plum
16. Pomegranate     17. Red-Grapefruit   18. Satsumas       19. Juice
20. Milk            21. Oatghurt         22. Oat-Milk       23. Sour-Cream
24. Sour-Milk       25. Soyghurt         26. Soy-Milk       27. Yoghurt
28. Asparagus       29. Aubergine        30. Cabbage        31. Carrots
32. Cucumber        33. Garlic           34. Ginger         35. Leek
36. Mushroom        37. Onion            38. Pepper         39. Potato
40. Red-Beet        41. Tomato           42. Zucchini
```

### Goals
#### Part 1: Design a Custom Neural Network
- Implement a convolutional neural network (CNN) for image classification from scratch.
- Aim to achieve around **60% validation accuracy**.
- Justify all design choices, including:
  - Network architecture
  - Training hyperparameters
  - Dataset preprocessing steps
- Document results and improvements using training plots, tables, or console outputs.

#### Part 2: Fine-Tune a Pretrained Network
- Fine-tune a **ResNet-18** model (pretrained on ImageNet-1K) using PyTorch.
- Fine-tuning steps:
  1. Train the model with the same hyperparameters used in Part 1.
  2. Optimize training hyperparameters to achieve a **validation accuracy between 80% and 90%**.



## Results
- **Part 1:** Achieved 62.16% accuracy with a custom CNN.
- **Part 2:** Achieved 88.51% accuracy by fine-tuning ResNet-18.


