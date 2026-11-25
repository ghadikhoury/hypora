export type ApprovalPolicy = "interactive" | "auto_within_contract";

export type TaskType =
  | "binary_classification"
  | "multiclass_classification"
  | "multilabel_classification"
  | "regression"
  | "forecasting"
  | "window_classification";

export type MetricName =
  | "accuracy"
  | "balanced_accuracy"
  | "precision"
  | "recall"
  | "f1"
  | "precision_macro"
  | "recall_macro"
  | "f1_macro"
  | "roc_auc"
  | "pr_auc"
  | "log_loss"
  | "brier_score"
  | "mae"
  | "rmse"
  | "r2";

export interface HealthResponse {
  status: string;
  service: string;
  db_path: string;
  db_exists: boolean;
  runs_root: string;
  git_sha: string | null;
  web_dist_ready: boolean;
}

export interface DatasetItem {
  name: string;
  source: string;
  dataset_id: number;
  title: string;
  version: string;
  license: string;
  target: string | null;
  rows: number;
  columns: number;
  file: string;
  path: string;
  exists: boolean;
}

export interface DatasetListResponse {
  snapshots_dir: string;
  manifest_exists: boolean;
  datasets: DatasetItem[];
}

export interface RunListItem {
  run_id: string;
  status: string;
  stage: string | null;
  dataset_label: string;
  goal: string | null;
  model_profile: string | null;
  approval_policy: string;
  budget_usd: number | null;
  spent_usd: number;
  budget_compute_s: number | null;
  spent_compute_s: number;
  pending_approval: boolean;
  final_verified: boolean | null;
  final_metric: string | null;
  final_score: number | null;
  source_db: string | null;
  read_only: boolean;
  created_at: string;
  updated_at: string;
}

export interface RunListResponse {
  runs: RunListItem[];
}

export interface RunCreateRequest {
  data: string;
  goal: string;
  target?: string | null;
  task_type?: TaskType | null;
  metric?: MetricName | null;
  forbid?: string[];
  budget_usd?: number | null;
  max_trials?: number | null;
  max_wall_clock_s?: number | null;
  approve?: ApprovalPolicy | null;
  advice_only?: boolean;
}

export interface RunCreateResponse {
  run_id: string;
  status: string;
  approval_policy: string;
}

export interface PendingApprovalDetail {
  approval_id: string;
  artifact_hash: string;
  hash_prefix: string;
  policy: string;
  summary: string;
  plan_version: number | null;
  artifact_ref: string | null;
  plan: Record<string, unknown> | null;
  scope: Record<string, unknown> | null;
}

export interface TrialMetricSplit {
  split: string;
  metrics: Record<string, unknown>;
  verification: Record<string, unknown>;
}

export interface TrialDetail {
  trial_id: string;
  kind: string;
  hypothesis: string;
  status: string;
  val_result: TrialMetricSplit | null;
  test_result: TrialMetricSplit | null;
}

export interface JobDetail {
  job_id: string;
  trial_id: string;
  status: string;
}

export interface ProfileSummary {
  record_id: string;
  dataset_ref: string;
  fingerprint: string;
  modality: string;
  tabular: Record<string, unknown> | null;
  vision: Record<string, unknown> | null;
  timeseries: Record<string, unknown> | null;
}

export interface RunDetailResponse {
  run_id: string;
  status: string;
  stage: string | null;
  dataset_label: string;
  approval_policy: string;
  read_only: boolean;
  source_db: string | null;
  budget_usd: number | NoneToUndefined;
  spent_usd: number;
  budget_compute_s: number | null;
  spent_compute_s: number;
  total_tokens: number;
  models_used: string[];
  model_config_snapshot: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  task_spec: Record<string, unknown> | null;
  profile: ProfileSummary | null;
  experiment_contract: Record<string, unknown> | null;
  data_contract: Record<string, unknown> | null;
  pending_approval: PendingApprovalDetail | null;
  trials: TrialDetail[];
  jobs: JobDetail[];
  report_json: Record<string, unknown> | null;
  report_md: string | null;
  cancel_requested: boolean;
}

type NoneToUndefined = null;

export interface TraceEventItem {
  seq: number;
  offset_s: number;
  created_at: string;
  role: string;
  type: string;
  duration_s: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  detail: string;
  trial_id: string | null;
  job_id: string | null;
  payload: Record<string, unknown>;
}

export interface TraceEventsResponse {
  run_id: string;
  status: string;
  events: TraceEventItem[];
  total_events: number;
  summary: string;
}

export interface ApprovalRequest {
  approve: boolean;
  hash_prefix: string;
  reason?: string | null;
  user?: string | null;
}

export interface ApprovalResponse {
  run_id: string;
  approval_id: string;
  decision: string;
  artifact_hash: string;
  decision_file: string;
}

export interface CancelResponse {
  run_id: string;
  status: string;
  cancel_requested: boolean;
  cancel_file: string;
}

export interface EvalSummaryResponse {
  consolidated_summary: Record<string, unknown> | null;
  evidence_index: Record<string, unknown> | null;
  f6_evidence: Record<string, unknown> | null;
  f7_evidence: Record<string, unknown> | null;
}