"""The harness that sends the sample to Laya and Jev and collects their answers.

- `questions`: the three questions from request.json, and how an answer is read back.
- `client`: one client for the `/v1/systemone` protocol, used for both systems.
- `runs`: resumable runs with raw logs, and the results table built from them.
- `truncation`: how many state tokens Laya cuts, which the strict server does not report.
- `training`: the training data for fine-tuning Laya on the DGX Spark, and installing the checkpoint it returns.
"""
