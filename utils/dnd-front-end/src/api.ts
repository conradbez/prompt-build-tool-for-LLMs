/** API client for the pbt server (proxied through Vite at /api → localhost:8000) */

const BASE = '/api';

export interface DagNodePayload {
  name: string;
  source: string;
}

export interface DagResponse {
  dag_id: string;
}

export interface RunResponse {
  outputs: Record<string, string>;
  errors: string[];
}

/** Register a DAG from the editor and get back a stable dag_id. */
export async function submitDag(nodes: DagNodePayload[]): Promise<DagResponse> {
  const res = await fetch(`${BASE}/dag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /dag failed (${res.status}): ${text}`);
  }
  return res.json();
}

/**
 * Run one or more models from a previously registered DAG.
 * Sent as multipart/form-data so promptfiles (File objects) can be included.
 */
export async function runDag(
  dagId: string,
  select?: string[],
  promptdata?: Record<string, string>,
  promptfiles?: Record<string, File>,
): Promise<RunResponse> {
  const form = new FormData();
  select?.forEach((s) => form.append('select', s));
  if (promptdata && Object.keys(promptdata).length > 0) {
    form.append('promptdata', JSON.stringify(promptdata));
  }
  if (promptfiles) {
    for (const [name, file] of Object.entries(promptfiles)) {
      // Use the key as filename — server strips extension to derive the promptfile key
      form.append('promptfiles', file, name);
    }
  }
  // Do NOT set Content-Type — browser sets multipart boundary automatically
  const res = await fetch(`${BASE}/dag/${dagId}/run`, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /dag/${dagId}/run failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<RunResponse>;
}
