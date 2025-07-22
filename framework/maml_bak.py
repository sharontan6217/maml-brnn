import torch
from torch import nn, optim

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# MAML Training Loop
def maml_train(model, tasks, meta_optimizer, inner_lr, meta_steps, inner_steps):
    for meta_step in range(meta_steps):
        meta_loss = 0.0
        for task in tasks:
            # Clone model for task-specific adaptation
            task_model = SimpleModel(*model.fc.weight.shape)
            task_model.load_state_dict(model.state_dict())
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            # Inner loop: task-specific adaptation
            for _ in range(inner_steps):
                task_loss = task['loss_fn'](task_model(task['data']), task['labels'])
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()

            # Compute meta-loss
            task_loss = task['loss_fn'](task_model(task['data']), task['labels'])
            meta_loss += task_loss

        # Meta-optimization step
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

# Example usage
input_size = 10
output_size = 1
model = SimpleModel(input_size, output_size)
meta_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy tasks for demonstration
tasks = [
    {'data': torch.randn(10, input_size), 'labels': torch.randn(10, output_size), 'loss_fn': nn.MSELoss()},
    {'data': torch.randn(10, input_size), 'labels': torch.randn(10, output_size), 'loss_fn': nn.MSELoss()}
]

# Train the model using MAML
maml_train(model, tasks, meta_optimizer, inner_lr=0.01, meta_steps=100, inner_steps=5)
