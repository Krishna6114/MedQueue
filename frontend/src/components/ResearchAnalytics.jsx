import React, { useEffect, useState } from 'react';
import { BarChart3, History, FileSpreadsheet, CheckCircle2 } from 'lucide-react';
import { medqueueApi } from '../api';

export default function ResearchAnalytics() {
  const [benchmarks, setBenchmarks] = useState(null);
  const [weightHistory, setWeightHistory] = useState([]);
  const [resolvedLog, setResolvedLog] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadAll() {
      setLoading(true);
      try {
        const [bData, wData, rData] = await Promise.all([
          medqueueApi.getBenchmarks().catch(() => null),
          medqueueApi.getWeightHistory().catch(() => []),
          medqueueApi.getResolvedHistory(25).catch(() => []),
        ]);
        setBenchmarks(bData);
        setWeightHistory(wData);
        setResolvedLog(rData);
      } finally {
        setLoading(false);
      }
    }
    loadAll();
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-slate-500 text-sm">Loading experimental benchmarks & audit logs...</div>;
  }

  return (
    <div className="space-y-8">
      {/* Live Session Weight Recalibration Log & Resolved Cases */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-3">
            <History className="w-4 h-4 text-teal-600" /> Live Session Weight Trajectory
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2">Cycle</th>
                  <th className="py-2">w₁</th>
                  <th className="py-2">w₂</th>
                  <th className="py-2">w₃</th>
                  <th className="py-2">Δw₁</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {weightHistory.map((row) => (
                  <tr key={row.cycle_index}>
                    <td className="py-2 font-semibold">#{row.cycle_index}</td>
                    <td className="py-2 font-mono text-teal-700 font-bold">{row.w1.toFixed(3)}</td>
                    <td className="py-2 font-mono">{row.w2.toFixed(3)}</td>
                    <td className="py-2 font-mono">{row.w3.toFixed(3)}</td>
                    <td className="py-2 font-mono">
                      {row.applied_delta_w1 >= 0 ? `+${row.applied_delta_w1.toFixed(4)}` : row.applied_delta_w1.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" /> Recently Resolved Consultations ({resolvedLog.length})
          </h3>
          {resolvedLog.length === 0 ? (
            <p className="text-xs text-slate-500 py-6 text-center">
              No consultations resolved in the live session yet. Complete cases in the Doctor Triage Dashboard to populate this audit trail.
            </p>
          ) : (
            <div className="overflow-x-auto max-h-64">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2">Case</th>
                    <th className="py-2">Patient</th>
                    <th className="py-2">Urgency</th>
                    <th className="py-2">Clinician Label</th>
                    <th className="py-2">Over-Flag Cause</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {resolvedLog.map((c) => (
                    <tr key={c.case_id}>
                      <td className="py-2 font-mono font-semibold">{c.case_id}</td>
                      <td className="py-2">{c.patient_name}</td>
                      <td className="py-2 font-semibold">{c.urgency_score.toFixed(1)}</td>
                      <td className="py-2">
                        <span className="px-2 py-0.5 rounded bg-slate-100 font-medium">
                          {c.feedback_label}
                        </span>
                      </td>
                      <td className="py-2 font-mono text-[11px] text-slate-600">
                        {c.over_flag_cause || 'none'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* 30-Run x 2,000-Case Simulation Benchmarks */}
      {benchmarks && (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-teal-600" />
              Monte Carlo Discrete-Event Simulation Benchmarks ({benchmarks.n_runs} Runs × {benchmarks.cases_per_run} Cases)
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Mean ± 95% confidence intervals across paired Poisson arrival runs (utilization ρ = 0.80, 12-point inversion tolerance).
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border border-slate-200 rounded-xl overflow-hidden">
              <thead className="bg-slate-100 text-slate-700 font-semibold">
                <tr>
                  <th className="p-3">Policy / Recalibration Condition</th>
                  <th className="p-3">High-Urgency Avg Wait (min)</th>
                  <th className="p-3">Inversion (Over-Flag) Rate</th>
                  <th className="p-3">Low-Urgency Max Wait (min)</th>
                  <th className="p-3">Low-Urgency 95th Pct (min)</th>
                  <th className="p-3">Mean Final w₁</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {benchmarks.summary_table.map((row) => (
                  <tr key={row.Condition} className="hover:bg-slate-50">
                    <td className="p-3 font-semibold text-slate-900">{row.Condition}</td>
                    <td className="p-3 font-mono">{row.High_Urgency_Wait_Min}</td>
                    <td className="p-3 font-mono">{row.Inversion_Rate}</td>
                    <td className="p-3 font-mono">{row.Low_Urgency_Max_Wait_Min}</td>
                    <td className="p-3 font-mono">{row.Low_Urgency_P95_Wait_Min}</td>
                    <td className="p-3 font-mono font-bold text-teal-700">{row.Mean_Final_w1.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Generated Publication Figures */}
          <div>
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-2 mb-4">
              <BarChart3 className="w-4 h-4 text-teal-600" /> Publication Benchmark Figures
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {benchmarks.figures.map((fig) => (
                <div key={fig.id} className="border border-slate-200 rounded-xl p-3 bg-slate-50">
                  <div className="text-xs font-semibold text-slate-700 mb-2">{fig.title}</div>
                  <img
                    src={fig.url}
                    alt={fig.title}
                    className="w-full h-auto rounded-lg bg-white border border-slate-100"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}