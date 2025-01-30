# Task Answers (Structured Data Processing)

(Last updated: Jan 29, 2025)

## Tutorial Tasks

```{eval-rst}
.. literalinclude:: util/answer.py
   :language: python
```

## PyTorch Implementation Tasks

The biggest problem is underfitting, which means that the model is too simple or the training procedure has problems, causing the model to be unable to catch the trend in the data. There could be many ways to increase performance. One example is gradually making the model more complex (e.g., by adding more layers or increasing the size of hidden units), tweaking the hyper-parameters (such as learning rate and weight decay), and observing model performance changes. Another possibility is to start with a complex model, try to overfit the data first, and then gradually reduce the model complexity. Below is the example set of model architecture and hyper-parameters. Notice that there could be multiple solutions to this problem.

Use the following model architecture with 3 layers and 512 hidden units:
```python
class DeepRegression(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        super(DeepRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out
```

During training, use the following hyper-parameters:
```python
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
```

Use 168 as the batch size:
```python
dataloader_train = DataLoader(dataset_train, batch_size=168, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=168, shuffle=False)
```

