# torchortho 📐  
[![arXiv](https://img.shields.io/badge/arXiv-2502.01247-b31b1b.svg)](https://arxiv.org/abs/2502.01247)
[![PyPI version](https://img.shields.io/pypi/v/torchortho)](https://pypi.org/project/torchortho/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/torchortho.svg)](https://pypi.org/project/torchortho/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learnable-polynomial-trigonometric-and/text-generation-on-openwebtext)](https://paperswithcode.com/sota/text-generation-on-openwebtext?p=learnable-polynomial-trigonometric-and)

`torchortho` is a **PyTorch library** for **learnable activation functions** based on:  
- **Hermite Polynomials** 🧙‍♂️
- **Fourier Series** 〰
- **Tropical Polynomials & Rational Functions** 🌴  

These **adaptive activations** dynamically adjust during training, offering improved expressivity, better gradient flow, and enhanced generalization for **vision and language models**.

---

## **📜 Paper Reference**
This library is based on the paper:  
📄 **[Learnable Polynomial, Trigonometric, and Tropical Activations](https://arxiv.org/abs/2502.01247)** *(Khalfaoui-Hassani & Kesselheim, 2025)*.  

For experimental results, check our repos:  
- **Vision models** (ConvNeXt with `torchortho` activations): [🔗 GitHub](https://github.com/K-H-Ismail/ConvNeXt-ortho)  
- **Language models** (GPT-2 with `torchortho` activations): [🔗 GitHub](https://github.com/K-H-Ismail/pytorch-language-models)  

---

## **📦 Installation**
Install from PyPI:  
```bash
pip install torchortho
```
or install directly from GitHub:
```bash
pip install git+https://github.com/K-H-Ismail/torchortho.git
```

---

## **📝 Usage**
You can use `torchortho` activations just like any other PyTorch activation:

### **Example: Using Hermite Activation**
```python
import torch
from torchortho import HermiteActivation

# Define a learnable Hermite activation
degree = 5
activation = HermiteActivation(degree)

# Forward pass
x = torch.rand(7, 4, 3, 2)
y = activation(x)

# Compute gradients
loss = y.sum()
loss.backward()

print("Gradients of activation coefficients:", activation.coefficients.grad)
print("Output:", y)
```

### **Example: Using Fourier Activation in a Neural Network**
```python
import torch
import torch.nn as nn
from torchortho import FourierActivation

class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = FourierActivation(degree=4)  # Learnable Fourier activation
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)

# Initialize the model
model = CustomMLP(input_dim=10, hidden_dim=32, output_dim=1)
x = torch.randn(5, 10)
output = model(x)
print("Model Output:", output)
```

---

## ⚡ Why Use torchortho?

### 1️⃣ Adaptive and Learnable Activations  
Unlike static activations (ReLU, GELU), `torchortho` functions **dynamically adapt** during training, allowing models to **learn optimal activation functions** for different tasks.

| Activation Type                 | Strengths |
|---------------------------------|-----------|
| **Hermite Activation**          | Adaptive polynomial approximation, variance-preserving, smooth optimization |
| **Fourier Activation**          | Captures periodic structures in data (useful for NLP, physics-based models, and time-series) |
| **Tropical Polynomial Activation** | Convex activation for structured learning (e.g., decision boundaries, optimization landscapes) |
| **Rational Activation**         | Generalizes standard activation functions (e.g., Pade approximants can replicate ReLU, GELU, or even SwiGLU) |

### 2️⃣ Improved Expressivity and Gradient Flow  
- **Better function approximation** → Increases expressivity for deep networks.  
- **Variance-preserving initialization** → Ensures stable training, avoiding vanishing/exploding gradients.  
- **More flexible than ReLU/SwiGLU** → Adapts activation behavior based on data.

### 3️⃣ Benchmarked on Real-World Models  
The effectiveness of `torchortho` activations has been validated on large-scale deep learning benchmarks:  

✅ **Image Classification (ConvNeXt-T on ImageNet-1K)**  
   - Replacing GELU with `torchortho` activations **improves top-1 accuracy**.  

✅ **Language Modeling (GPT-2 on OpenWebText)**  
   - Learnable activations **reduce perplexity** compared to GELU-based models.  

For full benchmarks, see:  
- **[Vision repo](https://github.com/K-H-Ismail/ConvNeXt-ortho)**  
- **[Language repo](https://github.com/K-H-Ismail/pytorch-language-models)**  


## **📜 License**
This project is licensed under the **GPL-3.0 License**. See [LICENSE](./LICENSE) for details.

---

## **🙌 Contributing**
We welcome contributions! Feel free to **submit issues, open PRs, or suggest improvements**.  

---

## **📬 Contact**
For questions or collaborations, reach out via **[GitHub Issues](https://github.com/K-H-Ismail/torchortho/issues)**.  

## 📚 Citation  
If you use `torchortho` in your research, please cite the following paper:

```bibtex
@article{khalfaoui2025learnable,
  title={Learnable polynomial, trigonometric, and tropical activations},
  author={Khalfaoui-Hassani, Ismail and Kesselheim, Stefan},
  journal={arXiv preprint arXiv:2502.01247},
  year={2025}
}
```
