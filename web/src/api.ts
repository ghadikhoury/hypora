import type {
  ApprovalRequest,
  ApprovalResponse,
  CancelResponse,
  DatasetListResponse,
  EvalSummaryResponse,
  HealthResponse,
  RunCreateRequest,
  RunCreateResponse,
  RunDetailResponse,
  RunListResponse,
  TraceEventItem,
  TraceEventsResponse,
} from "./types";

export class ApiClientError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const detail =
      typeof parsed === "object" && parsed !== null && "detail" in parsed
        ? (parsed as { detail: unknown }).detail
        : parsed;
    const msg =
      typeof detail === "string"
        ? detail
        : `HTTP ${res.status} on ${url}`;
    throw new ApiClientError(res.status, msg, detail);
  }
  return parsed as T;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function getDatasets(): Promise<DatasetListResponse> {
  return requestJson<DatasetListResponse>("/api/datasets");
}

export function getRuns(limit = 100): Promise<RunListResponse> {
  return requestJson<RunListResponse>(`/api/runs?limit=${limit}`);
}

export function createRun(payload: RunCreateRequest): Promise<RunCreateResponse> {
  return requestJson<RunCreateResponse>("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getRunDetail(runId: string): Promise<RunDetailResponse> {
  return requestJson<RunDetailResponse>(`/api/runs/${encodeURIComponent(runId)}`);
}

export interface EventQueryParams {
  type?: string;
  role?: string;
  trial?: string;
  job?: string;
  after_seq?: number;
  limit?: number;
}

export function getRunEvents(
  runId: string,
  params: EventQueryParams = {}
): Promise<TraceEventsResponse> {
  const qs = new URLSearchParams();
  if (params.type) qs.set("type", params.type);
  if (params.role) qs.set("role", params.role);
  if (params.trial) qs.set("trial", params.trial);
  if (params.job) qs.set("job", params.job);
  if (params.after_seq !== undefined) qs.set("after_seq", String(params.after_seq));
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return requestJson<TraceEventsResponse>(
    `/api/runs/${encodeURIComponent(runId)}/events${suffix}`
  );
}

export function approveRun(
  runId: string,
  payload: ApprovalRequest
): Promise<ApprovalResponse> {
  return requestJson<ApprovalResponse>(
    `/api/runs/${encodeURIComponent(runId)}/approve`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function cancelRun(runId: string): Promise<CancelResponse> {
  return requestJson<CancelResponse>(
    `/api/runs/${encodeURIComponent(runId)}/cancel`,
    {
      method: "POST",
    }
  );
}

export function getEvalSummary(): Promise<EvalSummaryResponse> {
  return requestJson<EvalSummaryResponse>("/api/eval/summary");
}

export interface StreamCallbacks {
  onTraceEvent: (ev: TraceEventItem) => void;
  onTerminal?: (info: { run_id: string; status: string }) => void;
  onError?: () => void;
}

export function subscribeRunEvents(
  runId: string,
  afterSeq: number,
  callbacks: StreamCallbacks
): () => void {
  const url = `/api/runs/${encodeURIComponent(runId)}/events/stream?after_seq=${afterSeq}`;
  const es = new EventSource(url);

  es.addEventListener("trace_event", (evt) => {
    try {
      const parsed = JSON.parse((evt as MessageEvent).data) as TraceEventItem;
      callbacks.onTraceEvent(parsed);
    } catch {
      // ignore malformed frame
    }
  });

  es.addEventListener("run_terminal", (evt) => {
    try {
      const parsed = JSON.parse((evt as MessageEvent).data) as {
        run_id: string;
        status: string;
      };
      callbacks.onTerminal?.(parsed);
    } catch {
      // ignore
    } finally {
      es.close();
    }
  });

  es.onerror = () => {
    callbacks.onError?.();
  };

  return () => {
    es.close();
  };
}