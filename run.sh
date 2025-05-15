#!/bin/bash

echo "Submitting workflow..."
# Submit the workflow and capture its generated name.
# The '-o name' flag ensures that only the workflow name is printed to stdout on success.
WORKFLOW_NAME=$(argo submit run.yaml --generate-name "my-workflow-" -n ci -o name)
SUBMIT_EXIT_CODE=$? # Capture the exit code of the argo submit command

# Check if 'argo submit' itself failed
if [ $SUBMIT_EXIT_CODE -ne 0 ]; then
  echo "Error: 'argo submit' command failed with exit code $SUBMIT_EXIT_CODE."
  # WORKFLOW_NAME variable might contain error output from argo submit in this case if it wrote to stdout.
  echo "Debug output from argo submit (if any): $WORKFLOW_NAME"
  exit 1 # Exit script with failure
fi

# Check if WORKFLOW_NAME is empty (e.g., if '-o name' produced no output even on exit code 0)
if [ -z "$WORKFLOW_NAME" ]; then
  echo "Error: 'argo submit' seemed to succeed but did not return a workflow name."
  exit 1 # Exit script with failure
fi

# Optional: A more robust check to ensure the workflow was actually created
# This helps catch edge cases where 'argo submit -o name' might succeed but the workflow isn't retrievable.
if ! argo get "$WORKFLOW_NAME" -n ci > /dev/null 2>&1; then
    echo "Error: Workflow '$WORKFLOW_NAME' was reported as submitted but cannot be found with 'argo get'."
    echo "This might indicate an issue with the submission or RBAC permissions."
    exit 1
fi

echo "Workflow submitted with generated name: $WORKFLOW_NAME"

echo "Following logs for $WORKFLOW_NAME..."
# This will stream logs. If you Ctrl+C here, the script will continue to the 'argo wait' part.
# If the workflow's logged pods complete, 'argo logs --follow' should also exit.
argo logs --follow "$WORKFLOW_NAME" -n ci
LOGS_FOLLOW_EXIT_CODE=$?

echo "Finished following logs (argo logs exit code: $LOGS_FOLLOW_EXIT_CODE)."
echo "Now waiting for workflow $WORKFLOW_NAME to complete and checking its final status..."

# Wait for the workflow to complete.
# `argo wait` will exit with 0 if the workflow Succeeded, and non-zero if it Failed or Errored.
if argo wait "$WORKFLOW_NAME" -n ci; then
  echo "Workflow $WORKFLOW_NAME completed successfully."

  exit 0 # Exit script with success
else
  WAIT_EXIT_CODE=$? # Capture the exit code of 'argo wait'
  echo "Error: Workflow $WORKFLOW_NAME did not complete successfully (argo wait exit code: $WAIT_EXIT_CODE)."

  ERROR_MESSAGE=$(argo get "$WORKFLOW_NAME" -n ci -o jsonpath='{.status.message}' 2>/dev/null || echo "N/A")

  echo "Final status: $FINAL_STATUS"
  echo "Message: $ERROR_MESSAGE"

  echo "Fetching last few lines of logs for potentially failed steps in $WORKFLOW_NAME..."
  # This provides a quick look at logs, especially if --follow was interrupted or didn't catch the error.
  argo logs "$WORKFLOW_NAME" -n ci --tail 50 # Show last 50 lines

  exit 1 # Exit script with failure
fi