import mlflow
import mlflow.pytorch
import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray (can connect to an existing cluster)
# ray.init(address="auto") # Use this to connect to a cluster
ray.init() # For local testing

# Define a remote function that logs to MLflow
@ray.remote
def train_model_and_log(X, y, tracking_uri, run_id):
    import mlflow # Import inside the function as it runs in a new process
    import mlflow.pytorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Ray task starting for run_id: {run_id}")

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Define a simple linear regression model
    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(1, 1)  # Input and output dimensions

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Set the tracking URI and start/get the existing run
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id) as run:
        logger.info("Inside MLflow run context in Ray task.")

        # Log parameters/metrics from within the task
        mlflow.log_param("ray_task_completed", "true")
        mlflow.log_metric("loss_from_task", loss.item())

        # Log system information
        mlflow.log_param("os_platform", os.name)
        mlflow.log_param("python_version", os.environ.get('PYTHON_VERSION', 'unknown'))
        mlflow.log_param("ray_version", ray.__version__)
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("numpy_version", np.__version__)

        # Example: Create a small artifact file from the task
        task_artifact_content = f"Task finished. Model weight: {model.linear.weight.item()}, bias: {model.linear.bias.item()}"
        with open("task_info.txt", "w") as f:
            f.write(task_artifact_content)
        mlflow.log_artifact("task_info.txt")
        logger.info("Logged task_info.txt artifact from Ray task.")

        # Log the model with an input example
        input_example = torch.tensor([[1.0]], dtype=torch.float32)
        # Convert PyTorch tensor to NumPy array
        input_example_np = input_example.clone().detach().numpy()
        mlflow.pytorch.log_model(model, "model_from_task", input_example=input_example_np)

    logger.info(f"Ray task finished for run_id: {run_id}")
    return model

# Create a simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize MLflow experiment and start a run on the client
mlflow.set_experiment("ray_mlflow_cde_demo")

# Get the current MLflow tracking URI and start the run
tracking_uri = mlflow.get_tracking_uri()
print(f"MLflow tracking URI: {tracking_uri}")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Started MLflow run with ID: {run_id}")

    # Log initial parameters on the client
    mlflow.log_param("dataset_size", len(X))

    # Log system information
    mlflow.log_param("os_platform", os.name)
    mlflow.log_param("python_version", os.environ.get('PYTHON_VERSION', 'unknown'))
    mlflow.log_param("ray_version", ray.__version__)
    mlflow.log_param("torch_version", torch.__version__)
    mlflow.log_param("numpy_version", np.__version__)

    # Submit the Ray task, passing tracking info
    model_ref = train_model_and_log.remote(X, y, tracking_uri, run_id)

    # Get the result (optional, depending on your workflow)
    model = ray.get(model_ref)

    # Test the model
    test_X = torch.tensor([[6], [7]], dtype=torch.float32)
    predictions = model(test_X)
    print(f"Predictions for test data: {predictions.detach().numpy()}")

    # Log metrics/parameters on the client after getting results
    test_y = np.array([12, 14]).reshape(-1, 1)
    test_loss = nn.MSELoss()(model(test_X.clone().detach()), 
                             torch.tensor(test_y, dtype=torch.float32))
    mlflow.log_metric("test_loss_client", test_loss.item())

    print("MLflow run completed successfully.")

# Shutdown Ray
ray.shutdown()
