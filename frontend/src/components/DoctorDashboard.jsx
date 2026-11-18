import React, { useState } from 'react';
import {
  Play,
  CheckCircle,
  ArrowUpCircle,
  ArrowDownCircle,
  Clock,
  Sliders,
  Users,
  FastForward,
  RotateCcw,
  Sparkles,
  AlertCircle,
} from 'lucide-react';
import { medqueueApi } from '../api';

export default function DoctorDashboard({ queueState, onRefresh }) {
  const [clinicianNote, setClinicianNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState(null);

  if (!queueState) return null;

  const {
    current_queue_minute,
    waiting_count,
    in_consultation_case,
    waiting_queue,
    weights,
    recalibration_mode,
    window_size,
    pending_feedback_count,
    completed_recalibration_cycles,
  } = queueState;

  const runAction = async (fn, successMsg = null) => {
    setBusy(true);
    setBanner(null);
    try {
      const res = await fn();
      if (successMsg) setBanner({ type: 'info', text: successMsg });
      await onRefresh();
      return res;
    } catch (err) {
      setBanner({ type: 'error', text: err.message });
    } finally {
      setBusy(false);
    }
  };

  const handleStartConsultation = (overrideId = null) =>
    runAction(() => medqueueApi.startConsultation(overrideId));

  const handleFeedback = async (label) => {
    if (!in_consultation_case) return;
    setBusy(true);
    setBanner(null);
    try {
      const res = await medqueueApi.submitFeedback({
        case_id: in_consultation_case.case_id,
        feedback_label: label,
        clinician_note: clinicianNote.trim(),
      });
      setClinicianNote('');
      if (res.recalibration_triggered && res.recalibration_event) {
        const ev = res.recalibration_event;
        setBanner({
          type: 'recal',
          text: `Recalibration Cycle #${ev.cycle_index} Triggered! w1 updated from ${ev.previous_weights.w1.toFixed(3)} to ${ev.updated_weights.w1.toFixed(3)} (Δw1 = ${ev.applied_delta_w1 >= 0 ? '+' : ''}${ev.applied_delta_w1.toFixed(4)}).`,
        });
      } else {
        setBanner({
          type: 'info',
          text: `Recorded "${label.replace('_', ' ')}" for ${res.case_id} (${res.pending_feedback_count}/${res.window_size} cases in active window).`,
        });
      }
      await onRefresh();
    } catch (err) {
      setBanner({ type: 'error', text: err.message });
    } finally {
      setBusy(false);
    }
  };

  const handleModeChange = (newMode) =>
    runAction(
      () => medqueueApi.updateConfig({ recalibration_mode: newMode, window_size: window_size, delta: 0.08 }),
      `Switched recalibration mode to "${newMode}".`
    );

  const urgencyColor = (u) => {
    if (u >= 75) return 'bg-rose-100 text-rose-800 border-rose-300';
    if (u >= 45) return 'bg-amber-100 text-amber-800 border-amber-300';
    return 'bg-emerald-100 text-emerald-800 border-emerald-300';
  };

  return (
    <div className="space-y-6">
      {/* Top Control & Weight Telemetry Bar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Adaptive Weights Card */}
        <div className="lg:col-span-7 bg-white rounded-2xl shadow-sm border border-slate-200 p-4 flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-teal-600" />
              <span className="text-sm font-bold text-slate-800">
                Composite Ranking Weights: R_i(t) = w₁·u_i + w₂·(t − t_i) + w₃·(100·c_i)
              </span>
            </div>
            <span className="text-xs font-mono bg-slate-100 text-slate-700 px-2.5 py-1 rounded-lg">
              Clock: {current_queue_minute.toFixed(1)} min
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="bg-teal-50 border border-teal-200 rounded-xl p-2.5 text-center">
              <div className="text-[11px] font-semibold text-teal-700">Urgency Weight (w₁)</div>
              <div className="text-xl font-extrabold text-teal-900">{weights.w1.toFixed(3)}</div>
            </div>
            <div className="bg-sky-50 border border-sky-200 rounded-xl p-2.5 text-center">
              <div className="text-[11px] font-semibold text-sky-700">Wait Escalation (w₂)</div>
              <div className="text-xl font-extrabold text-sky-900">{weights.w2.toFixed(3)}</div>
            </div>
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-2.5 text-center">
              <div className="text-[11px] font-semibold text-indigo-700">Confidence (w₃)</div>
              <div className="text-xl font-extrabold text-indigo-900">{weights.w3.toFixed(3)}</div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-500 font-medium">Update Rule:</span>
              <select
                value={recalibration_mode}
                onChange={(e) => handleModeChange(e.target.value)}
                disabled={busy}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 font-medium text-slate-700 focus:outline-none"
              >
                <option value="root_cause_aware">Refined Option A (Root-Cause-Aware)</option>
                <option value="baseline_paper">Baseline Paper Eq. 3</option>
                <option value="fixed">Fixed Weights (No Recalibration)</option>
              </select>
            </div>
            <div className="text-slate-600 font-medium">
              Review Window: <span className="font-bold text-teal-700">{pending_feedback_count} / {window_size}</span>{' '}
              (Cycles completed: {completed_recalibration_cycles})
            </div>
          </div>
        </div>

        {/* Live Demo & Simulation Controls */}
        <div className="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-800">Live Simulation & Demo Controls</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Seed synthetic ESI-1 to ESI-5 arrivals or advance clock to observe wait-time starvation prevention.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-4">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                runAction(() => medqueueApi.seedBatch(10, 2.0), 'Seeded 10 stratified synthetic patient cases.')
              }
              className="flex flex-col items-center justify-center gap-1 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold py-2.5 px-2 rounded-xl transition cursor-pointer disabled:opacity-50"
            >
              <Users className="w-4 h-4 text-teal-400" />
              <span>Seed 10 Cases</span>
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                runAction(() => medqueueApi.advanceTime(30.0), 'Advanced queue clock by +30.0 minutes.')
              }
              className="flex flex-col items-center justify-center gap-1 bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold py-2.5 px-2 rounded-xl transition cursor-pointer disabled:opacity-50"
            >
              <FastForward className="w-4 h-4" />
              <span>+30m Clock</span>
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => runAction(() => medqueueApi.resetSystem(), 'System queue and weights reset to initial state.')}
              className="flex flex-col items-center justify-center gap-1 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 text-xs font-semibold py-2.5 px-2 rounded-xl transition cursor-pointer disabled:opacity-50"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset All</span>
            </button>
          </div>
        </div>
      </div>

      {/* Feedback / Status Banner */}
      {banner && (
        <div
          className={`p-3.5 rounded-xl border text-sm flex items-center gap-2.5 ${
            banner.type === 'error'
              ? 'bg-rose-50 border-rose-200 text-rose-800'
              : banner.type === 'recal'
              ? 'bg-teal-50 border-teal-300 text-teal-900 font-semibold'
              : 'bg-sky-50 border-sky-200 text-sky-900'
          }`}
        >
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{banner.text}</span>
        </div>
      )}

      {/* Main Split: Active Consultation Panel (Left) + Ranked Priority Queue (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Active Consultation & 3-Label Clinician Feedback */}
        <div className="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-slate-900">Active Consultation Room</h3>
            {in_consultation_case ? (
              <span className="text-xs font-semibold bg-teal-100 text-teal-800 px-2.5 py-1 rounded-full">
                In Session: {in_consultation_case.case_id}
              </span>
            ) : (
              <span className="text-xs font-medium bg-slate-100 text-slate-500 px-2.5 py-1 rounded-full">
                Room Available
              </span>
            )}
          </div>

          {in_consultation_case ? (
            <div className="space-y-4">
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-bold text-slate-900 text-base">
                      {in_consultation_case.patient_name}
                      {in_consultation_case.patient_age ? `, ${in_consultation_case.patient_age}y` : ''}
                    </div>
                    <div className="text-xs text-slate-500">
                      Waited {in_consultation_case.elapsed_wait.toFixed(1)} min · Composite R(t) ={' '}
                      <strong>{in_consultation_case.composite_score.toFixed(1)}</strong>
                    </div>
                  </div>
                  <span
                    className={`text-xs font-bold px-2.5 py-1 rounded-lg border ${urgencyColor(
                      in_consultation_case.urgency_score
                    )}`}
                  >
                    u = {in_consultation_case.urgency_score.toFixed(1)}
                  </span>
                </div>
                <p className="text-sm text-slate-800 mt-3 leading-relaxed">
                  "{in_consultation_case.symptom_text}"
                </p>
              </div>

              {/* Token XAI */}
              <div>
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5 flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5 text-teal-600" /> Explainability Drivers
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {in_consultation_case.top_contributors.map((t) => (
                    <span
                      key={t.token}
                      className={`text-xs font-mono px-2 py-1 rounded-md border ${
                        t.contribution >= 0
                          ? 'bg-rose-50 border-rose-200 text-rose-800'
                          : 'bg-emerald-50 border-emerald-200 text-emerald-800'
                      }`}
                    >
                      {t.token} ({t.contribution >= 0 ? '+' : ''}
                      {t.contribution.toFixed(1)})
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-600 mb-1">
                  Optional Clinical / Triage Note
                </label>
                <input
                  type="text"
                  value={clinicianNote}
                  onChange={(e) => setClinicianNote(e.target.value)}
                  placeholder="Optional observation for audit log..."
                  className="w-full rounded-xl border border-slate-300 px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              {/* Structured 3-Label Clinician Feedback Buttons */}
              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs font-semibold text-slate-700 mb-2.5">
                  Complete Consultation & Rate Queue Prioritization:
                </p>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => handleFeedback('correctly_prioritized')}
                    className="flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold py-2.5 px-4 rounded-xl transition cursor-pointer disabled:opacity-50"
                  >
                    <CheckCircle className="w-4 h-4" />
                    <span>Correctly Prioritized</span>
                  </button>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleFeedback('over_flagged')}
                      className="flex items-center justify-center gap-1.5 bg-amber-500 hover:bg-amber-600 text-white text-xs font-semibold py-2.5 px-3 rounded-xl transition cursor-pointer disabled:opacity-50"
                    >
                      <ArrowDownCircle className="w-4 h-4" />
                      <span>Over-Flagged</span>
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleFeedback('under_flagged')}
                      className="flex items-center justify-center gap-1.5 bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold py-2.5 px-3 rounded-xl transition cursor-pointer disabled:opacity-50"
                    >
                      <ArrowUpCircle className="w-4 h-4" />
                      <span>Under-Flagged</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="border border-dashed border-slate-300 rounded-xl p-8 text-center space-y-3">
              <p className="text-sm text-slate-500">
                No patient currently in consultation. Click below to call the #1 ranked patient from the adaptive priority queue.
              </p>
              <button
                type="button"
                disabled={busy || waiting_count === 0}
                onClick={() => handleStartConsultation(null)}
                className="inline-flex items-center gap-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition cursor-pointer"
              >
                <Play className="w-4 h-4" />
                <span>Consult Next Highest-Ranked Patient</span>
              </button>
            </div>
          )}
        </div>

        {/* Live Ranked Priority Queue */}
        <div className="lg:col-span-7 bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Live Priority Queue ({waiting_count} Waiting)
              </h3>
              <p className="text-xs text-slate-500">
                Dynamically re-ranked every cycle by composite score R_i(t). Clinicians may override any rank.
              </p>
            </div>
          </div>

          {waiting_queue.length === 0 ? (
            <div className="border border-dashed border-slate-300 rounded-xl p-10 text-center text-sm text-slate-500">
              Queue is empty. Submit a patient in the <strong>Patient Intake</strong> tab or click{' '}
              <strong>Seed 10 Cases</strong> above.
            </div>
          ) : (
            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {waiting_queue.map((item) => {
                const totalR = Math.max(item.composite_score, 1);
                const uPct = (item.term_breakdown.urgency_term / totalR) * 100;
                const wPct = (item.term_breakdown.wait_term / totalR) * 100;
                const cPct = (item.term_breakdown.confidence_term / totalR) * 100;

                return (
                  <div
                    key={item.case_id}
                    className={`rounded-xl border p-4 transition ${
                      item.rank_position === 1
                        ? 'border-teal-400 bg-teal-50/30 shadow-xs'
                        : 'border-slate-200 bg-white hover:border-slate-300'
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-extrabold ${
                            item.rank_position === 1
                              ? 'bg-teal-600 text-white'
                              : 'bg-slate-100 text-slate-700'
                          }`}
                        >
                          #{item.rank_position}
                        </span>
                        <div>
                          <div className="font-bold text-sm text-slate-900">
                            {item.patient_name}{' '}
                            <span className="font-mono text-xs font-normal text-slate-400">
                              ({item.case_id})
                            </span>
                          </div>
                          <div className="flex items-center gap-3 text-xs text-slate-500 mt-0.5">
                            <span className="inline-flex items-center gap-1">
                              <Clock className="w-3 h-3" /> Waited {item.elapsed_wait.toFixed(1)} min
                            </span>
                            <span>Conf: {(item.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span
                          className={`text-xs font-bold px-2.5 py-1 rounded-lg border ${urgencyColor(
                            item.urgency_score
                          )}`}
                        >
                          Urgency: {item.urgency_score.toFixed(1)}
                        </span>
                        <span className="text-xs font-extrabold bg-slate-900 text-white px-2.5 py-1 rounded-lg">
                          R(t) = {item.composite_score.toFixed(1)}
                        </span>
                        {!in_consultation_case && (
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => handleStartConsultation(item.case_id)}
                            title="Clinician Override: Consult this patient immediately"
                            className="text-xs font-semibold text-teal-700 hover:text-teal-900 bg-teal-50 hover:bg-teal-100 border border-teal-200 px-2.5 py-1 rounded-lg transition cursor-pointer"
                          >
                            {item.rank_position === 1 ? 'Consult' : 'Override'}
                          </button>
                        )}
                      </div>
                    </div>

                    <p className="text-xs text-slate-700 mt-2.5 line-clamp-2">{item.symptom_text}</p>

                    {/* Composite Term Contribution Bar */}
                    <div className="mt-3">
                      <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                        <span>Urgency term: {item.term_breakdown.urgency_term.toFixed(1)}</span>
                        <span>Wait term: {item.term_breakdown.wait_term.toFixed(1)}</span>
                        <span>Conf term: {item.term_breakdown.confidence_term.toFixed(1)}</span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden flex">
                        <div style={{ width: `${uPct}%` }} className="bg-teal-500" title="Urgency term" />
                        <div style={{ width: `${wPct}%` }} className="bg-sky-500" title="Wait escalation term" />
                        <div style={{ width: `${cPct}%` }} className="bg-indigo-400" title="Confidence term" />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}