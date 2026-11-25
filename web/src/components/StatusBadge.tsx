import React from "react";

interface StatusBadgeProps {
  value: string;
}

const FOREST_VALUES = new Set([
  "DONE",
  "SUCCEEDED",
  "PASSED",
  "VERIFIED",
  "ALLOWED",
  "APPROVED",
  "OK",
  "FIRST_HIGHER",
]);

const AMBER_VALUES = new Set([
  "WAITING_APPROVAL",
  "INCONCLUSIVE",
  "PENDING",
  "QUEUED",
  "CANCEL_REQUESTED",
  "WARNING",
]);

const CRIMSON_VALUES = new Set([
  "FAILED",
  "FAILED_TERMINAL",
  "DENIED",
  "BLOCKED",
  "ERROR",
  "CANCELLED",
  "TIMED_OUT",
  "REJECTED",
]);

const COBALT_VALUES = new Set([
  "INTAKE",
  "PHASE_A",
  "PHASE_B",
  "REPORTING",
  "RUNNING",
  "ACTIVE",
  "CONTRACT",
  "PROFILE",
]);

export const StatusBadge: React.FC<StatusBadgeProps> = ({ value }) => {
  const norm = value.toUpperCase();
  let cls = "badge";
  if (FOREST_VALUES.has(norm)) {
    cls += " badge-forest";
  } else if (AMBER_VALUES.has(norm)) {
    cls += " badge-amber";
  } else if (CRIMSON_VALUES.has(norm)) {
    cls += " badge-crimson";
  } else if (COBALT_VALUES.has(norm)) {
    cls += " badge-cobalt";
  }
  return <span className={cls}>{value}</span>;
};