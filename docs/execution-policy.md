# Execution Policy

## Hard requirement

This project is cloud-run only. The local user device must not execute production computation.

Production computation includes:
- historical backtests and execution-constrained backtests;
- walk-forward research and parameter sensitivity;
- data backfills and production data refreshes;
- candidate and portfolio generation;
- scheduled production calculations.

## Allowed local-device actions

The local device may only be used for:
- editing files;
- inspecting files and logs;
- reading GitHub results;
- triggering or monitoring GitHub Actions.

The local device must not be used to launch Python, Node, PowerShell, or other commands that perform project computation.

## Cloud execution standard

Production jobs must run through GitHub Actions on an approved GitHub-hosted runner. The current standard runner is ubuntu-24.04.

Long-running jobs must be initiated through a GitHub Actions workflow, not through Remote Desktop Commander.

## Result authority

GitHub is the execution authority and source of truth for production research outputs. Locally generated research artifacts are not production results and must not be committed as production evidence.

## Current constrained-backtest workflow

.github/workflows/historical-execution-constraints.yml is the canonical cloud entry point for the execution-constrained historical backtest. It checks out the canonical private strategy, runs the backtest on GitHub Actions, validates the result, and commits the resulting audit artifacts back to main.
