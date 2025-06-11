## 📁 Project Structure

This project includes three main folders:

### 🔹 `V1_greedy`
Implements a **greedy allocation algorithm**, where container assignment is based on predefined rules.  
In this version, **RL** is used to fine-tune the weight parameters for better performance.

### 🔹 `V2`
A copy of `V1_greedy`, currently being modified to explore other **heuristic algorithms** for container allocation.  
This version is under development for experimenting with alternative rule-based strategies.

### 🔹 `V3_RL`
Implements a full **DRL** approach for container allocation.  
It includes an **attention-based network** to enhance the model’s decision-making.  
The `test_order` module in this folder compares the performance of the DRL model with the `V1_greedy` method.
