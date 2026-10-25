---
prompt: single_agent_final
version: 1
---
You have finished calling tools. Return your final answer as JSON:

- `final_trial_id`: the `trial_id` of the trial you recommend as the final model, chosen from the
  trials whose experiments completed; null if no experiment completed.
- `confidence`: how sure you are that it is the best of your trials, from 0 to 1.
