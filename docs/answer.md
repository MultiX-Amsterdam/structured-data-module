# Task Answers (Structured Data Processing)

(Last updated: Sep 8, 2025)

## Tutorial Tasks

```{eval-rst}
.. literalinclude:: util/answer.py
   :language: python
```

### Answer for Task 7
- Q1: In your opinion, is including sensor data in the past a good idea to help improve model performance?
- Q2: In your opinion, is the wind direction from the air quality monitoring station (i.e., feed 28) that near the pollution source a good feature to predict bad smell?

Based on the experiment, we have the following result:
| Experiment | Feature Set     | Model   | F1-Score  | Precision | Recall   |
|------------|-----------------|---------|-----------|-----------|----------|
| E1         | S1              | DT      | 0.32      | 0.46      | 0.27     |
| E2         | S2 (+wind)      | DT      | 0.37      | 0.40      | 0.37     |
| E3         | S3 (+hour)      | DT      | 0.33      | 0.40      | 0.31     |
| E4         | S4 (+hour+wind) | DT      | 0.38      | 0.41      | 0.38     |
| E5         | S1              | RF      | 0.34      | 0.46      | 0.26     |
| E6         | S2 (+wind)      | RF      | 0.38      | 0.46      | 0.35     |
| E7         | S3 (+hour)      | RF      | 0.32      | 0.46      | 0.27     |
| E8         | S4 (+hour+wind) | RF      | 0.39      | 0.55      | 0.34     |

Model DT means decision tree, and RF means random forest. The f1-score, precision, and recall are averaged for all the cross-validation folds. Also, the numbers in the table could be different than the ones in the tutorial. The reason is that there is randomness in the training process of both models, so the results are going to be a little bit different each time.

For question Q1, we can see that the performance does not really increase when we add more data from the previous 2 hours (by comparing experiment pairs E1/E3, E2/E4, E5/E7, and E6/E8). However, more experiments are needed to determine the effect of adding historical data. It could be the reason that the pollution took more hours to travel from the source to the city region. So we really need to add more data from the previous hours to see the effect, for example, by also including the data from the previous 3 and 4 hours.

For question Q2, we already see 4% to 7% performance increases when adding wind data (by comparing experiment pairs E1/E2, E3/E4, E5/E6, E7/E8). This suggests that wind data indeed plays an important role in smell prediction, which makes sense as the wind direction and speed affect how the pollution travels to the city region from the source in a real-world situation.


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
