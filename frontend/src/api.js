const API_BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errData = await res.json();
      if (errData.detail) detail = errData.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const medqueueApi = {
  getHealth: () => request('/health'),
  getQueue: () => request('/queue'),
  submitIntake: (payload) =>
    request('/intake', { method: 'POST', body: JSON.stringify(payload) }),
  startConsultation: (overrideCaseId = null) =>
    request('/consultation/start', {
      method: 'POST',
      body: JSON.stringify({ override_case_id: overrideCaseId }),
    }),
  submitFeedback: (payload) =>
    request('/consultation/feedback', { method: 'POST', body: JSON.stringify(payload) }),
  getWeightHistory: () => request('/history/weights'),
  getResolvedHistory: (limit = 30) => request(`/history/resolved?limit=${limit}`),
  getBenchmarks: () => request('/benchmarks'),
  updateConfig: (payload) =>
    request('/config/recalibration', { method: 'POST', body: JSON.stringify(payload) }),
  seedBatch: (count = 10, staggerMinutes = 2.0) =>
    request('/demo/seed-batch', {
      method: 'POST',
      body: JSON.stringify({ count, stagger_minutes: staggerMinutes }),
    }),
  advanceTime: (minutes = 15.0) =>
    request('/demo/advance-time', {
      method: 'POST',
      body: JSON.stringify({ minutes }),
    }),
  resetSystem: () => request('/demo/reset', { method: 'POST' }),
};