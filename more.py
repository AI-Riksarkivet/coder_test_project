import mlflow
import mlflow.pytorch
import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper function to get Git information (from previous response) ---
def get_git_info():
    # ... (implementation from previous response) ...
    git_info = {}
    try:
        git_info["git_commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        git_info["git_branch"] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        # Try to get repo URL, may fail if not in a repo or no remote named origin
        try:
            git_info["git_repo_url"] = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            git_info["git_repo_url"] = "unknown"
        dirty_status = subprocess.check_output(['git', 'status', '--porcelain']).strip().decode('utf-8')
        git_info["git_dirty"] = "true" if dirty_status else "false"
    except subprocess.CalledProcessError:
        logger.warning("Could not retrieve full Git information. Is this a git repository with a remote 'origin'?")
        git_info.setdefault("git_commit", "unknown")
        git_info.setdefault("git_branch", "unknown")
        git_info.setdefault("git_repo_url", "unknown")
        git_info.setdefault("git_dirty", "unknown")
    return git_info

# Initialize Ray
ray.init() # For local testing

# Define a remote function that logs to MLflow
# ... (train_model_and_log function from your original script, potentially with task-specific OS/lib versions)
@ray.remote
def train_model_and_log(X, y, tracking_uri, run_id):
    import mlflow
    import mlflow.pytorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import logging
    import os
    import platform as task_platform # Alias to avoid conflict
    import numpy as np # Ensure numpy is available
    import ray as task_ray # Alias

    logger_task = logging.getLogger(__name__ + ".task")
    logger_task.info(f"Ray task starting for run_id: {run_id}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(1, 1)
        def forward(self, x):
            return self.linear(x)

    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id) as run:
        logger_task.info("Inside MLflow run context in Ray task.")
        mlflow.log_param("ray_task_completed", "true")
        mlflow.log_metric("loss_from_task", loss.item())

        # Log system information from the Ray worker
        mlflow.log_param("task_os_platform", task_platform.platform())
        mlflow.log_param("task_python_version", task_platform.python_version())
        mlflow.log_param("task_node_hostname", task_platform.node())
        mlflow.log_param("task_ray_version", task_ray.__version__)
        mlflow.log_param("task_torch_version", torch.__version__)
        mlflow.log_param("task_numpy_version", np.__version__)

        task_artifact_content = f"Task finished. Model weight: {model.linear.weight.item()}, bias: {model.linear.bias.item()}"
        with open("task_info.txt", "w") as f:
            f.write(task_artifact_content)
        mlflow.log_artifact("task_info.txt", artifact_path="ray_task_artifacts")
        logger_task.info("Logged task_info.txt artifact from Ray task.")

        input_example_np = torch.tensor([[1.0]], dtype=torch.float32).clone().detach().numpy()
        mlflow.pytorch.log_model(model, "model_from_task", input_example=input_example_np)

    logger_task.info(f"Ray task finished for run_id: {run_id}")
    return model


# --- Main client script ---
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

EXPERIMENT_NAME = "ray_mlflow_coder_demo" # Updated experiment name
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

tracking_uri = mlflow.get_tracking_uri() # This will be from MLFLOW_TRACKING_URI env var if set by Coder
print(f"MLflow tracking URI: {tracking_uri}")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print(f"Started MLflow run with ID: {run_id} in experiment ID: {experiment_id}")

    # --- Log Standard Workspace Information (Client/Driver) ---
    client_workspace_info = {}
    client_workspace_info["run_id"] = run_id
    client_workspace_info["experiment_id"] = experiment_id
    client_workspace_info["mlflow_tracking_uri_resolved"] = tracking_uri # Resolved URI by client
    # Get script path robustly
    try:
        client_workspace_info["script_path"] = str(Path(__file__).resolve())
    except NameError: # If __file__ is not defined (e.g. interactive)
        client_workspace_info["script_path"] = sys.argv[0] if sys.argv else "unknown_script"

    client_workspace_info["working_directory"] = os.getcwd()
    try:
        client_workspace_info["user"] = os.getlogin()
    except OSError: # os.getlogin() can fail in some non-interactive environments
        client_workspace_info["user"] = os.environ.get("USER", os.environ.get("LOGNAME", "unknown_user"))

    client_workspace_info["client_hostname"] = platform.node() # This will be the K8s Pod Name/ID
    client_workspace_info["client_os_platform"] = platform.platform()
    client_workspace_info["client_python_version"] = platform.python_version()
    client_workspace_info["client_ray_version"] = ray.__version__
    client_workspace_info["client_torch_version"] = torch.__version__
    client_workspace_info["client_numpy_version"] = np.__version__
    client_workspace_info["client_mlflow_version"] = mlflow.__version__

    # Git information
    git_info = get_git_info()
    client_workspace_info.update(git_info)
    mlflow.log_params(client_workspace_info)

    # --- Log Coder & Kubernetes Specific Information ---
    coder_kube_info = {}
    # Standard Coder env vars
    for var in ["CODER_WORKSPACE_NAME", "CODER_WORKSPACE_ID", "CODER_ENVIRONMENT_NAME",
                "CODER_ENVIRONMENT_ID", "CODER_USER_NAME", "CODER_USER_EMAIL", "CODER_USER_ID",
                "CODER_AGENT_NAME", "MLFLOW_TRACKING_URI", "ARGO_BASE_HREF"]:
        value = os.environ.get(var)
        if value:
            coder_kube_info[var.lower()] = value

    # Coder parameters (assuming Terraform maps them to these env var names)
    # You MUST ensure your Terraform deployment section passes these as env vars
    # For example, data.coder_parameter.cpu.value -> CODER_PARAM_CPU
    coder_param_map = {
        "coder_param_cpu": "CODER_PARAM_CPU",
        "coder_param_memory_gb": "CODER_PARAM_MEMORY_GB", # Assuming you add GB suffix in TF or use as is
        "coder_param_home_disk_gb": "CODER_PARAM_HOME_DISK_GB",
        "coder_param_gpu_type": "CODER_PARAM_GPU_TYPE",
        "coder_param_gpu_count": "CODER_PARAM_GPU_COUNT", # The one selected by user
        "coder_actual_gpu_count": "CODER_ACTUAL_GPU_COUNT", # From local.actual_gpu_count
        "coder_k8s_namespace": "CODER_K8S_NAMESPACE", # Pass var.namespace
        "coder_container_image": "CODER_CONTAINER_IMAGE" # Pass the image string
    }
    for key_mlflow, env_var_name in coder_param_map.items():
        value = os.environ.get(env_var_name)
        if value:
            coder_kube_info[key_mlflow] = value
    
    # If CODER_CONTAINER_IMAGE wasn't set, log the known static one
    if "coder_container_image" not in coder_kube_info:
        coder_kube_info["coder_container_image_static"] = "registry.ra.se:5002/devenv:v6.0.0"


    # Dockerfile static info (can be logged directly as they are known from Dockerfile context)
    coder_kube_info["docker_base_image"] = "ubuntu:jammy"
    coder_kube_info["docker_cuda_version"] = "12.2"
    coder_kube_info["docker_python_version_target"] = "3.12"
    coder_kube_info["docker_venv_path"] = "/opt/venv-py312"
    coder_kube_info["lakefs_endpoint_config"] = "http://lakefs.lakefs:80/"


    if coder_kube_info:
        mlflow.log_params(coder_kube_info)

    mlflow.log_param("initial_dataset_size", len(X))

    # Log Ray cluster info (basic for local Ray within the Pod)
    try:
        cluster_resources = ray.cluster_resources()
        mlflow.log_params({
            "ray_pod_cpus_available": cluster_resources.get("CPU", 0),
            "ray_pod_gpus_available": cluster_resources.get("GPU", 0), # Will reflect GPUs allocated to pod
            "ray_pod_memory_gb_available": round(cluster_resources.get("memory", 0) / (1024**3), 2)
        })
    except Exception as e:
        logger.warning(f"Could not log Ray pod resources: {e}")

    # Log requirements as an artifact (if pip is available and you want current env)
    # Note: Your Dockerfile already installs specific versions, this would be for any ad-hoc additions
    try:
        requirements_path = Path("requirements_runtime.txt")
        with open(requirements_path, "w") as f:
            # Using uv if available (as per your Dockerfile) or pip
            freeze_cmd = "uv pip freeze" if subprocess.run("command -v uv", shell=True, capture_output=True).returncode == 0 else "pip freeze"
            subprocess.call(freeze_cmd.split(), stdout=f)
        mlflow.log_artifact(str(requirements_path), artifact_path="environment_setup")
    except Exception as e:
        logger.warning(f"Could not log runtime requirements.txt: {e}")

    # Submit the Ray task
    model_ref = train_model_and_log.remote(X, y, tracking_uri, run_id)
    model = ray.get(model_ref)

    # Test the model
    test_X = torch.tensor([[6], [7]], dtype=torch.float32)
    predictions = model(test_X)
    print(f"Predictions for test data: {predictions.detach().numpy()}")

    test_y = np.array([12, 14]).reshape(-1, 1)
    test_loss = nn.MSELoss()(model(test_X.clone().detach()),
                             torch.tensor(test_y, dtype=torch.float32))
    mlflow.log_metric("test_loss_client", test_loss.item())

    mlflow.set_tag("run_status", "completed_successfully_in_coder")
    print("MLflow run completed successfully in Coder workspace.")

ray.shutdown()