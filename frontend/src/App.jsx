import React, { useCallback, useEffect, useState } from 'react';
import { Activity, UserPlus, Stethoscope, LineChart } from 'lucide-react';
import { medqueueApi } from './api';
import PatientIntake from './components/PatientIntake';
import DoctorDashboard from './components/DoctorDashboard';
import ResearchAnalytics from './components/ResearchAnalytics';

export default function App() {
  const [activeTab, setActiveTab] = useState('doctor');
  const [queueState, setQueueState] = useState(null);
  const [apiError, setApiError] = useState('');

  const refreshQueue = useCallback(async () => {
    try {
      const data = await medqueueApi.getQueue();
      setQueueState(data);
      setApiError('');
    } catch (err) {
      setApiError('Unable to reach MedQueue FastAPI backend at http://127.0.0.1:8000. Ensure Uvicorn is running.');
    }
  }, []);

  useEffect(() => {
    refreshQueue();
    const timer = setInterval(refreshQueue, 10000);
    return () => clearInterval(timer);
  }, [refreshQueue]);

  const navItems = [
    { id: 'doctor', label: 'Doctor Triage Dashboard', icon: Stethoscope },
    { id: 'intake', label: 'Patient Intake Portal', icon: UserPlus },
    { id: 'analytics', label: 'Research & Simulation Lab', icon: LineChart },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header Navigation */}
      <header className="bg-slate-900 text-white border-b border-slate-800 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-teal-500/20 border border-teal-400/40 flex items-center justify-center">
              <Activity className="w-5 h-5 text-teal-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold tracking-tight">MedQueue</h1>
                <span className="text-[11px] font-semibold bg-teal-500/20 text-teal-300 border border-teal-500/30 px-2 py-0.5 rounded-full">
                  Adaptive Triage v1.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Clinician-Feedback-Driven Recalibration & Starvation-Preventing Priority Queue
              </p>
            </div>
          </div>

          <nav className="flex items-center gap-1.5 bg-slate-800/80 p-1 rounded-xl">
            {navItems.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                    isActive
                      ? 'bg-teal-600 text-white shadow-xs'
                      : 'text-slate-300 hover:text-white hover:bg-slate-700/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                  {tab.id === 'doctor' && queueState && (
                    <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] bg-slate-900/60 text-teal-200">
                      {queueState.waiting_count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {apiError && (
          <div className="mb-5 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm font-medium">
            {apiError}
          </div>
        )}

        {activeTab === 'doctor' && <DoctorDashboard queueState={queueState} onRefresh={refreshQueue} />}
        {activeTab === 'intake' && (
          <PatientIntake
            onSubmitted={() => {
              refreshQueue();
            }}
          />
        )}
        {activeTab === 'analytics' && <ResearchAnalytics />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-4 text-center text-xs text-slate-500">
        MedQueue Research Prototype · Queue Prioritization Decision Support Only (Non-Diagnostic) · Clinician Retains Full Override Authority
      </footer>
    </div>
  );
}