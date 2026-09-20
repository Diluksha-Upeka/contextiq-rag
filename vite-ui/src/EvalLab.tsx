import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  AlertTriangle,
  Play,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  Zap,
  Target,
  FileQuestion,
  ChevronDown,
  ChevronUp,
  Plus,
  Trash2,
  Clock,
  TrendingUp,
} from 'lucide-react';


const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

export type RetrievalMetrics = {
  recall_at_k: number;
  precision_at_k: number;
  mrr: number;
  ndcg_at_k: number;
};

export type GenerationMetrics = {
  faithfulness: number;
  answer_relevance: number;
  context_relevance: number;
  unsupported_claims?: string[];
  faithfulness_reasoning?: string;
};

export type QuestionResult = {
  id: string;
  question: string;
  category: string;
  expected_answer: string;
  actual_answer: string;
  latency_ms: number;
  retrieved_count: number;
  retrieved_pages: number[];
  retrieval_metrics: RetrievalMetrics;
  generation_metrics: GenerationMetrics;
};

export type EvalRunResult = {
  id: string;
  config: {
    name: string;
    retrieval_mode: string;
    top_k: number;
    confidence_threshold: number;
    run_llm_judge: boolean;
  };
  timestamp: string;
  total_questions: number;
  duration_ms: number;
  average_metrics: {
    recall_at_k: number;
    precision_at_k: number;
    mrr: number;
    ndcg_at_k: number;
    faithfulness: number;
    answer_relevance: number;
    context_relevance: number;
  };
  question_results: QuestionResult[];
};

export type DatasetQuestion = {
  id: string;
  question: string;
  expected_answer: string;
  category: string;
  relevant_pages: number[];
  relevant_keywords: string[];
};

export type Dataset = {
  name: string;
  description: string;
  document_name: string;
  questions: DatasetQuestion[];
};

export type ComparisonResponse = {
  timestamp: string;
  document_name: string;
  total_questions: number;
  configurations: {
    name: string;
    mode: string;
    duration_ms: number;
    metrics: {
      recall_at_k: number;
      precision_at_k: number;
      mrr: number;
      ndcg_at_k: number;
      faithfulness: number;
      answer_relevance: number;
      context_relevance: number;
    };
  }[];
  full_runs: EvalRunResult[];
};

export default function EvalLab({ onBackToChat }: { onBackToChat: () => void }) {
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'comparison' | 'dataset'>('benchmarks');
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [resultsHistory, setResultsHistory] = useState<EvalRunResult[]>([]);
  const [selectedRun, setSelectedRun] = useState<EvalRunResult | null>(null);
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null);

  // Configuration form state
  const [retrievalMode, setRetrievalMode] = useState<string>('hybrid_rerank');
  const [topK, setTopK] = useState<number>(5);
  const [runJudge, setRunJudge] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [comparingLoading, setComparingLoading] = useState<boolean>(false);
  const [expandedQuestions, setExpandedQuestions] = useState<Record<string, boolean>>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // New question state
  const [newQText, setNewQText] = useState('');
  const [newQExpected, setNewQExpected] = useState('');
  const [newQCategory, setNewQCategory] = useState('factual');
  const [newQPages, setNewQPages] = useState('1');
  const [newQKeywords, setNewQKeywords] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);

  // Load initial dataset & history
  useEffect(() => {
    fetchDataset();
    fetchHistory();
  }, []);

  const fetchDataset = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/eval/dataset`);
      if (res.ok) {
        const data = await res.json();
        setDataset(data);
      }
    } catch (err) {
      console.error('Failed to fetch dataset', err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/eval/results`);
      if (res.ok) {
        const history: EvalRunResult[] = await res.json();
        setResultsHistory(history);
        if (history.length > 0 && !selectedRun) {
          setSelectedRun(history[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch results history', err);
    }
  };

  const handleRunEval = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const configName =
        retrievalMode === 'dense'
          ? 'Baseline (Dense Only)'
          : retrievalMode === 'bm25'
          ? 'BM25 Keyword Only'
          : retrievalMode === 'hybrid'
          ? 'Hybrid (Dense + BM25)'
          : 'Hybrid + Cross-Encoder Reranker';

      const res = await fetch(`${API_BASE_URL}/api/eval/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          namespace: 'latest',
          config_name: configName,
          retrieval_mode: retrievalMode,
          top_k: topK,
          confidence_threshold: 0.35,
          run_llm_judge: runJudge,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Evaluation failed: ${errText}`);
      }

      const runData: EvalRunResult = await res.json();
      setSelectedRun(runData);
      setResultsHistory((prev) => [runData, ...prev]);
      setActiveTab('benchmarks');
    } catch (err: any) {
      setErrorMsg(err.message || 'Evaluation run failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompareAll = async () => {
    setComparingLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/eval/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          namespace: 'latest',
          top_k: topK,
          run_llm_judge: runJudge,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Comparison failed: ${errText}`);
      }

      const data: ComparisonResponse = await res.json();
      setComparisonData(data);
      setActiveTab('comparison');
    } catch (err: any) {
      setErrorMsg(err.message || 'Comparison failed');
    } finally {
      setComparingLoading(false);
    }
  };

  const handleAddQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dataset || !newQText.trim() || !newQExpected.trim()) return;

    const pages = newQPages
      .split(',')
      .map((p) => parseInt(p.trim(), 10))
      .filter((p) => !isNaN(p));

    const keywords = newQKeywords
      .split(',')
      .map((k) => k.trim())
      .filter(Boolean);

    const newQuestion: DatasetQuestion = {
      id: `q${dataset.questions.length + 1}`,
      question: newQText.trim(),
      expected_answer: newQExpected.trim(),
      category: newQCategory,
      relevant_pages: pages,
      relevant_keywords: keywords,
    };

    const updatedDataset: Dataset = {
      ...dataset,
      questions: [...dataset.questions, newQuestion],
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/eval/dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedDataset),
      });

      if (res.ok) {
        setDataset(updatedDataset);
        setShowAddModal(false);
        setNewQText('');
        setNewQExpected('');
        setNewQPages('1');
        setNewQKeywords('');
      }
    } catch (err) {
      console.error('Failed to save question', err);
    }
  };

  const handleDeleteQuestion = async (qid: string) => {
    if (!dataset) return;
    const updated: Dataset = {
      ...dataset,
      questions: dataset.questions.filter((q) => q.id !== qid),
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/eval/dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated),
      });
      if (res.ok) {
        setDataset(updated);
      }
    } catch (err) {
      console.error('Failed to delete question', err);
    }
  };

  const toggleQuestionExpanded = (id: string) => {
    setExpandedQuestions((prev) => ({ ...prev, [id]: !prev[id] }));
  };


  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-white tracking-tight">ContextIQ Evaluation Lab</h1>
              <span className="text-[10px] uppercase font-mono tracking-wider px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">
                RAG Benchmarking
              </span>
            </div>
            <p className="text-xs text-slate-400">Measure IR retrieval recall, precision, and LLM grounding metrics</p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Navigation Tabs */}
          <div className="flex bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'benchmarks'
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Benchmark Runs
            </button>
            <button
              onClick={() => setActiveTab('comparison')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'comparison'
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TrendingUp className="w-3.5 h-3.5" />
              Pipeline Comparison
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'dataset'
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileQuestion className="w-3.5 h-3.5" />
              Golden Dataset ({dataset?.questions?.length ?? 0})
            </button>
          </div>

          <button
            onClick={onBackToChat}
            className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition flex items-center gap-1.5 shadow-sm"
          >
            ← Return to Chat
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {errorMsg && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-3 text-rose-400 text-sm">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <p>{errorMsg}</p>
          </div>
        )}

        {/* Control Bar: Pipeline Configuration & Execution */}
        <section className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div>
                <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                  Retrieval Pipeline
                </label>
                <select
                  value={retrievalMode}
                  onChange={(e) => setRetrievalMode(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="dense">Baseline (Dense Vector Only)</option>
                  <option value="bm25">BM25 Okapi (Keyword Only)</option>
                  <option value="hybrid">Hybrid (Dense + BM25 RRF)</option>
                  <option value="hybrid_rerank">Hybrid + Cross-Encoder Reranker</option>
                </select>
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                  Top-K Depth
                </label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                  className="bg-slate-800 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={3}>Top 3</option>
                  <option value={5}>Top 5 (Recommended)</option>
                  <option value={8}>Top 8</option>
                  <option value={10}>Top 10</option>
                </select>
              </div>

              <div className="flex items-center gap-2 pt-4">
                <input
                  type="checkbox"
                  id="judgeToggle"
                  checked={runJudge}
                  onChange={(e) => setRunJudge(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="judgeToggle" className="text-xs text-slate-300 font-medium cursor-pointer">
                  Run Gemini LLM Judge (Faithfulness & Relevance)
                </label>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleCompareAll}
                disabled={comparingLoading || isLoading}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 transition flex items-center gap-2 disabled:opacity-50"
              >
                {comparingLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    Comparing Pipelines...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-3.5 h-3.5" />
                    Compare All 3 Pipelines
                  </>
                )}
              </button>

              <button
                onClick={handleRunEval}
                disabled={isLoading || comparingLoading}
                className="px-5 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/25 transition flex items-center gap-2 disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    Evaluating ({dataset?.questions?.length || 0} queries)...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Run Benchmark
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* TAB 1: Benchmark Run Details */}
        {activeTab === 'benchmarks' && (
          <div className="space-y-6">
            {selectedRun ? (
              <>
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                  {/* Recall@K */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">Recall@{selectedRun.config.top_k}</span>
                      <Target className="w-4 h-4 text-blue-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {Math.round(selectedRun.average_metrics.recall_at_k * 100)}%
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Ground-truth chunk coverage</p>
                  </div>

                  {/* Precision@K */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">Precision@{selectedRun.config.top_k}</span>
                      <Zap className="w-4 h-4 text-indigo-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {Math.round(selectedRun.average_metrics.precision_at_k * 100)}%
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Relevant chunk ratio</p>
                  </div>

                  {/* MRR */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">MRR</span>
                      <BarChart3 className="w-4 h-4 text-emerald-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {selectedRun.average_metrics.mrr.toFixed(3)}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Reciprocal rank of 1st hit</p>
                  </div>

                  {/* nDCG */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">nDCG@{selectedRun.config.top_k}</span>
                      <TrendingUp className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {selectedRun.average_metrics.ndcg_at_k.toFixed(3)}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Ranking order quality</p>
                  </div>

                  {/* Faithfulness */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">Faithfulness</span>
                      <ShieldCheck className="w-4 h-4 text-teal-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {selectedRun.config.run_llm_judge
                        ? `${Math.round(selectedRun.average_metrics.faithfulness * 100)}%`
                        : 'N/A'}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Context grounded claims</p>
                  </div>

                  {/* Answer Relevance */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">Answer Rel</span>
                      <Sparkles className="w-4 h-4 text-amber-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {selectedRun.config.run_llm_judge
                        ? `${Math.round(selectedRun.average_metrics.answer_relevance * 100)}%`
                        : 'N/A'}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Direct query alignment</p>
                  </div>

                  {/* Latency */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-sm">
                    <div className="flex items-center justify-between text-slate-400 mb-1">
                      <span className="text-[11px] font-semibold uppercase tracking-wider">Total Time</span>
                      <Clock className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="text-2xl font-bold text-white tracking-tight">
                      {(selectedRun.duration_ms / 1000).toFixed(1)}s
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">{selectedRun.total_questions} queries batch</p>
                  </div>
                </div>

                {/* Per-Question Detailed Breakdown */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                  <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                    <div>
                      <h2 className="text-sm font-bold text-white">Evaluation Question Breakdown</h2>
                      <p className="text-xs text-slate-400">
                        {selectedRun.config.name} • {selectedRun.timestamp}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      {resultsHistory.length > 1 && (
                        <select
                          value={selectedRun.id}
                          onChange={(e) => {
                            const found = resultsHistory.find((r) => r.id === e.target.value);
                            if (found) setSelectedRun(found);
                          }}
                          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-lg px-2 py-1"
                        >
                          {resultsHistory.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.config.name} ({r.timestamp})
                            </option>
                          ))}
                        </select>
                      )}
                      <span className="text-xs px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
                        {selectedRun.question_results.length} questions
                      </span>
                    </div>
                  </div>


                  <div className="divide-y divide-slate-800/60">
                    {selectedRun.question_results.map((q) => {
                      const isExpanded = expandedQuestions[q.id];
                      return (
                        <div key={q.id} className="p-5 hover:bg-slate-850/40 transition">
                          <div
                            onClick={() => toggleQuestionExpanded(q.id)}
                            className="cursor-pointer flex items-start justify-between gap-4"
                          >
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-mono font-bold text-blue-400 uppercase">
                                  {q.id}
                                </span>
                                <span
                                  className={`text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full ${
                                    q.category === 'out_of_domain'
                                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                      : 'bg-slate-800 text-slate-300 border border-slate-700'
                                  }`}
                                >
                                  {q.category}
                                </span>
                                <span className="text-xs text-slate-500">• {q.latency_ms}ms</span>
                              </div>
                              <p className="text-sm font-medium text-slate-100">{q.question}</p>
                            </div>

                            {/* Mini score badges */}
                            <div className="flex items-center gap-2 flex-shrink-0">
                              <div className="text-right">
                                <span className="text-[10px] text-slate-400 block">Recall / Prec</span>
                                <span className="text-xs font-mono font-semibold text-slate-200">
                                  {Math.round(q.retrieval_metrics.recall_at_k * 100)}% /{' '}
                                  {Math.round(q.retrieval_metrics.precision_at_k * 100)}%
                                </span>
                              </div>

                              {selectedRun.config.run_llm_judge && (
                                <div className="text-right pl-3 border-l border-slate-800">
                                  <span className="text-[10px] text-slate-400 block">Faithfulness</span>
                                  <span
                                    className={`text-xs font-mono font-semibold ${
                                      q.generation_metrics.faithfulness >= 0.8
                                        ? 'text-emerald-400'
                                        : 'text-rose-400'
                                    }`}
                                  >
                                    {Math.round(q.generation_metrics.faithfulness * 100)}%
                                  </span>
                                </div>
                              )}

                              <button className="p-1 text-slate-400 hover:text-white transition ml-2">
                                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                              </button>
                            </div>
                          </div>

                          {/* Expanded Details */}
                          {isExpanded && (
                            <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-3 text-xs">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                                  <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block mb-1">
                                    Expected Ground Truth
                                  </span>
                                  <p className="text-slate-300 leading-relaxed">{q.expected_answer}</p>
                                </div>

                                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                                  <span className="text-[10px] uppercase font-bold text-blue-400 tracking-wider block mb-1">
                                    ContextIQ Generated Answer
                                  </span>
                                  <p className="text-slate-200 leading-relaxed">{q.actual_answer}</p>
                                </div>
                              </div>

                              {/* Faithfulness / unsupported claims alert if any */}
                              {q.generation_metrics.unsupported_claims &&
                                q.generation_metrics.unsupported_claims.length > 0 && (
                                  <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300">
                                    <span className="font-semibold block mb-0.5">
                                      ⚠️ Hallucinated / Unsupported Claims Detected:
                                    </span>
                                    <ul className="list-disc list-inside space-y-0.5 text-amber-200/90">
                                      {q.generation_metrics.unsupported_claims.map((claim, idx) => (
                                        <li key={idx}>{claim}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                              <div className="flex items-center justify-between text-slate-400 text-[11px] pt-1">
                                <span>
                                  Retrieved Pages:{' '}
                                  <span className="text-slate-200 font-mono">
                                    {q.retrieved_pages.length > 0 ? q.retrieved_pages.join(', ') : 'None'}
                                  </span>
                                </span>
                                <span>
                                  MRR: <span className="text-slate-200 font-mono">{q.retrieval_metrics.mrr}</span> | nDCG:{' '}
                                  <span className="text-slate-200 font-mono">{q.retrieval_metrics.ndcg_at_k}</span>
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-20 bg-slate-900/40 rounded-2xl border border-slate-800 space-y-3">
                <FileQuestion className="w-12 h-12 text-slate-600 mx-auto" />
                <h3 className="text-base font-semibold text-slate-300">No Benchmark Runs Yet</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Click <strong>Run Benchmark</strong> above to evaluate the current document index against the golden
                  dataset.
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Side-by-Side Comparison */}
        {activeTab === 'comparison' && (
          <div className="space-y-6">
            {comparisonData ? (
              <div className="space-y-6">
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
                  <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-800">
                    <div>
                      <h2 className="text-base font-bold text-white">Pipeline Architecture Benchmark Comparison</h2>
                      <p className="text-xs text-slate-400">
                        Evaluated across {comparisonData.total_questions} ground-truth test questions on document:{' '}
                        <span className="text-blue-400 font-medium">{comparisonData.document_name}</span>
                      </p>
                    </div>
                    <span className="text-xs px-3 py-1 rounded-full bg-blue-500/20 text-blue-300 font-mono border border-blue-500/30">
                      {comparisonData.timestamp}
                    </span>
                  </div>

                  {/* Comparative Metrics Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] tracking-wider">
                          <th className="py-3 px-4 font-semibold">Configuration</th>
                          <th className="py-3 px-4 font-semibold">Recall@5</th>
                          <th className="py-3 px-4 font-semibold">Precision@5</th>
                          <th className="py-3 px-4 font-semibold">MRR</th>
                          <th className="py-3 px-4 font-semibold">nDCG@5</th>
                          <th className="py-3 px-4 font-semibold">Faithfulness</th>
                          <th className="py-3 px-4 font-semibold">Latency</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono">
                        {comparisonData.configurations.map((cfg, idx) => {
                          const isBest = idx === comparisonData.configurations.length - 1;
                          return (
                            <tr
                              key={idx}
                              className={`hover:bg-slate-800/40 transition ${
                                isBest ? 'bg-blue-600/10 font-bold text-white' : 'text-slate-300'
                              }`}
                            >
                              <td className="py-4 px-4 font-sans font-medium flex items-center gap-2">
                                {cfg.name}
                                {isBest && (
                                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                    Highest Quality
                                  </span>
                                )}
                              </td>
                              <td className="py-4 px-4">{Math.round(cfg.metrics.recall_at_k * 100)}%</td>
                              <td className="py-4 px-4">{Math.round(cfg.metrics.precision_at_k * 100)}%</td>
                              <td className="py-4 px-4">{cfg.metrics.mrr.toFixed(3)}</td>
                              <td className="py-4 px-4">{cfg.metrics.ndcg_at_k.toFixed(3)}</td>
                              <td className="py-4 px-4">
                                {cfg.metrics.faithfulness > 0
                                  ? `${Math.round(cfg.metrics.faithfulness * 100)}%`
                                  : 'N/A'}
                              </td>
                              <td className="py-4 px-4 text-slate-400">{(cfg.duration_ms / 1000).toFixed(1)}s</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Visual Comparative Bars */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Target className="w-4 h-4 text-blue-400" />
                      Recall@5 Comparison (Coverage)
                    </h3>
                    <div className="space-y-3">
                      {comparisonData.configurations.map((c, i) => {
                        const pct = Math.round(c.metrics.recall_at_k * 100);
                        return (
                          <div key={i} className="space-y-1">
                            <div className="flex justify-between text-xs text-slate-300">
                              <span>{c.name}</span>
                              <span className="font-mono font-bold text-blue-400">{pct}%</span>
                            </div>
                            <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  i === 2
                                    ? 'bg-gradient-to-r from-blue-500 to-indigo-500'
                                    : i === 1
                                    ? 'bg-indigo-600'
                                    : 'bg-slate-600'
                                }`}
                                style={{ width: `${Math.max(pct, 5)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-emerald-400" />
                      nDCG Ranking Quality Comparison
                    </h3>
                    <div className="space-y-3">
                      {comparisonData.configurations.map((c, i) => {
                        const score = c.metrics.ndcg_at_k;
                        const pct = Math.round(score * 100);
                        return (
                          <div key={i} className="space-y-1">
                            <div className="flex justify-between text-xs text-slate-300">
                              <span>{c.name}</span>
                              <span className="font-mono font-bold text-emerald-400">{score.toFixed(3)}</span>
                            </div>
                            <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  i === 2
                                    ? 'bg-gradient-to-r from-emerald-500 to-teal-500'
                                    : i === 1
                                    ? 'bg-teal-600'
                                    : 'bg-slate-600'
                                }`}
                                style={{ width: `${Math.max(pct, 5)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-20 bg-slate-900/40 rounded-2xl border border-slate-800 space-y-3">
                <TrendingUp className="w-12 h-12 text-slate-600 mx-auto" />
                <h3 className="text-base font-semibold text-slate-300">No Comparison Data Yet</h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  Click <strong>Compare All 3 Pipelines</strong> to evaluate Baseline Dense, Hybrid BM25+Dense, and
                  Hybrid+Cross-Encoder reranking side-by-side!
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Golden Dataset Manager */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
              <div>
                <h2 className="text-sm font-bold text-white">Golden Evaluation Dataset</h2>
                <p className="text-xs text-slate-400">
                  Curated test queries with expected answers, target pages, and relevant keywords for benchmarking
                </p>
              </div>
              <button
                onClick={() => setShowAddModal(true)}
                className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white transition flex items-center gap-1.5 shadow"
              >
                <Plus className="w-4 h-4" />
                Add Question
              </button>
            </div>

            {/* Questions List */}
            <div className="space-y-3">
              {dataset?.questions?.map((q, idx) => (
                <div
                  key={q.id}
                  className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex items-start justify-between gap-4"
                >
                  <div className="space-y-2 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold text-blue-400">#{idx + 1} ({q.id})</span>
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 uppercase">
                        {q.category}
                      </span>
                      {q.relevant_pages?.length > 0 && (
                        <span className="text-xs text-slate-400">
                          Target Pages: <span className="text-slate-200 font-mono">{q.relevant_pages.join(', ')}</span>
                        </span>
                      )}
                    </div>
                    <p className="text-sm font-medium text-white">{q.question}</p>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      <strong className="text-slate-300">Expected:</strong> {q.expected_answer}
                    </p>
                    {q.relevant_keywords?.length > 0 && (
                      <div className="flex items-center gap-1.5 pt-1">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">Keywords:</span>
                        {q.relevant_keywords.map((kw, i) => (
                          <span
                            key={i}
                            className="text-[10px] px-2 py-0.5 rounded bg-slate-800/80 text-slate-400 border border-slate-700/60"
                          >
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <button
                    onClick={() => handleDeleteQuestion(q.id)}
                    className="p-2 text-slate-500 hover:text-rose-400 transition"
                    title="Delete question"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Modal: Add New Question */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-white">Add Benchmark Question</h3>

            <form onSubmit={handleAddQuestion} className="space-y-3.5">
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Question</label>
                <input
                  type="text"
                  required
                  value={newQText}
                  onChange={(e) => setNewQText(e.target.value)}
                  placeholder="e.g. What is the key advantage of RRF?"
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Expected Answer</label>
                <textarea
                  required
                  rows={3}
                  value={newQExpected}
                  onChange={(e) => setNewQExpected(e.target.value)}
                  placeholder="Ideal ground truth answer expected from the system..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Category</label>
                  <select
                    value={newQCategory}
                    onChange={(e) => setNewQCategory(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="factual">Factual</option>
                    <option value="technical">Technical</option>
                    <option value="analysis">Analysis</option>
                    <option value="out_of_domain">Out of Domain (Unanswerable)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">Relevant Pages (comma-separated)</label>
                  <input
                    type="text"
                    value={newQPages}
                    onChange={(e) => setNewQPages(e.target.value)}
                    placeholder="1, 2"
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1">Relevant Keywords (comma-separated)</label>
                <input
                  type="text"
                  value={newQKeywords}
                  onChange={(e) => setNewQKeywords(e.target.value)}
                  placeholder="fusion, ranking, reciprocal"
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow"
                >
                  Save Question
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
