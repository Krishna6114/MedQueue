import React, { useState } from 'react';
import { Send, AlertTriangle, CheckCircle2, Sparkles, ShieldAlert } from 'lucide-react';
import { medqueueApi } from '../api';

const SAMPLE_PRESETS = [
  {
    tier: 'ESI-1 (Critical)',
    color: 'border-rose-300 bg-rose-50 text-rose-800 hover:bg-rose-100',
    name: 'Raghavendra Rao',
    age: 62,
    text: 'crushing chest pain radiating to left arm and jaw, accompanied by profuse sweating and pallor, started 15 minutes ago.',
  },
  {
    tier: 'ESI-2 (Emergent)',
    color: 'border-amber-300 bg-amber-50 text-amber-800 hover:bg-amber-100',
    name: 'Meenakshi Iyer',
    age: 45,
    text: 'high fever 104F with rigors confusion and rapid breathing, unable to keep fluids down, for the past 4 hours.',
  },
  {
    tier: 'ESI-3 (Urgent)',
    color: 'border-sky-300 bg-sky-50 text-sky-800 hover:bg-sky-100',
    name: 'Karan Malhotra',
    age: 33,
    text: 'moderate fever 101.5F with productive cough and body aches, feeling fatigued but alert and breathing comfortably, for 2 days.',
  },
  {
    tier: 'ESI-5 (Routine)',
    color: 'border-emerald-300 bg-emerald-50 text-emerald-800 hover:bg-emerald-100',
    name: 'Sneha Kulkarni',
    age: 29,
    text: 'routine prescription refill request for stable blood pressure medication, feeling completely well with no new symptoms, scheduled monthly check.',
  },
];

export default function PatientIntake({ onSubmitted }) {
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [symptomText, setSymptomText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const applyPreset = (preset) => {
    setPatientName(preset.name);
    setPatientAge(String(preset.age));
    setSymptomText(preset.text);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!symptomText.trim()) return;
    setSubmitting(true);
    setError('');
    try {
      const data = await medqueueApi.submitIntake({
        patient_name: patientName.trim() || 'Patient',
        patient_age: patientAge ? Number(patientAge) : null,
        symptom_text: symptomText.trim(),
      });
      setResult(data);
      if (onSubmitted) onSubmitted();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const urgencyBadge = (score) => {
    if (score >= 75) return 'bg-rose-100 text-rose-800 border-rose-300';
    if (score >= 45) return 'bg-amber-100 text-amber-800 border-amber-300';
    return 'bg-emerald-100 text-emerald-800 border-emerald-300';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Intake Form */}
      <div className="lg:col-span-7 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-900">Asynchronous Tele-Consultation Intake</h2>
            <p className="text-sm text-slate-500 mt-1">
              Describe symptoms in plain language. MedQueue estimates urgency and uncertainty solely to prioritize queue order.
            </p>
          </div>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3.5 mb-5 flex items-start gap-3 text-xs text-amber-900">
          <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong>Non-Diagnostic Queueing Notice:</strong> This tool assists consultation scheduling only and does not provide medical diagnoses. If you are experiencing a life-threatening emergency, call emergency services immediately.
          </div>
        </div>

        {/* Quick Presets */}
        <div className="mb-5">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
            Quick Clinical Test Presets
          </p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_PRESETS.map((p) => (
              <button
                key={p.tier}
                type="button"
                onClick={() => applyPreset(p)}
                className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition cursor-pointer ${p.color}`}
              >
                {p.tier}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sm:col-span-2">
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient Full Name</label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g., Aarav Sharma"
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Age (Years)</label>
              <input
                type="number"
                min="0"
                max="120"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g., 42"
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Symptom Description <span className="text-rose-500">*</span>
            </label>
            <textarea
              rows={4}
              required
              value={symptomText}
              onChange={(e) => setSymptomText(e.target.value)}
              placeholder="Describe primary symptoms, severity, associated signs (fever, sweating, breathlessness), and duration..."
              className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
            />
          </div>

          {error && (
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !symptomText.trim()}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-xl transition cursor-pointer"
          >
            <Send className="w-4 h-4" />
            {submitting ? 'Scoring & Enqueueing...' : 'Submit to Priority Queue'}
          </button>
        </form>
      </div>

      {/* Live Scoring & XAI Preview Card */}
      <div className="lg:col-span-5">
        {result ? (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-5">
            <div className="flex items-center justify-between">
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 bg-teal-50 px-3 py-1 rounded-full">
                <CheckCircle2 className="w-4 h-4" /> Enqueued ({result.case_id})
              </span>
              <span className="text-xs text-slate-500">Arrival t = {result.arrival_minute} min</span>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className={`p-3 rounded-xl border ${urgencyBadge(result.urgency_score)}`}>
                <div className="text-xs font-medium opacity-80">Urgency (u_i)</div>
                <div className="text-2xl font-extrabold mt-0.5">{result.urgency_score.toFixed(1)}</div>
                <div className="text-[11px] opacity-75">out of 100</div>
              </div>
              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <div className="text-xs font-medium text-slate-500">Confidence (c_i)</div>
                <div className="text-2xl font-extrabold text-slate-800 mt-0.5">
                  {(result.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-[11px] text-slate-500">ensemble agreement</div>
              </div>
              <div className="p-3 rounded-xl border border-indigo-200 bg-indigo-50">
                <div className="text-xs font-medium text-indigo-600">Queue Position</div>
                <div className="text-2xl font-extrabold text-indigo-900 mt-0.5">
                  #{result.current_rank_position}
                </div>
                <div className="text-[11px] text-indigo-600">of {result.total_waiting} waiting</div>
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-teal-600" /> Token-Level Urgency Attribution (XAI)
              </h3>
              <p className="text-xs text-slate-600 bg-slate-50 p-3 rounded-xl border border-slate-200 mb-3">
                {result.explanation_summary}
              </p>
              <div className="space-y-2">
                {result.top_contributors.map((item) => (
                  <div key={item.token} className="flex items-center justify-between text-xs bg-slate-50 px-3 py-2 rounded-lg border border-slate-100">
                    <span className="font-mono font-semibold text-slate-800">"{item.token}"</span>
                    <span
                      className={`font-semibold px-2 py-0.5 rounded ${
                        item.contribution >= 0
                          ? 'bg-rose-100 text-rose-700'
                          : 'bg-emerald-100 text-emerald-700'
                      }`}
                    >
                      {item.contribution >= 0 ? `+${item.contribution.toFixed(2)}` : item.contribution.toFixed(2)} pts
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-sm border border-dashed border-slate-300 p-8 text-center flex flex-col items-center justify-center h-full min-h-[320px]">
            <Sparkles className="w-10 h-10 text-teal-500 mb-3 opacity-80" />
            <h3 className="text-base font-semibold text-slate-800">Real-Time Urgency & Explainability</h3>
            <p className="text-sm text-slate-500 mt-1 max-w-sm">
              Submit a patient case on the left (or select an ESI preset) to view its TF-IDF + Bagged Ridge urgency score, ensemble confidence, and token attributions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}