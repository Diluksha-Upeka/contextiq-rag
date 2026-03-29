import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, Loader2, Info, Bot, User, BookOpen } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

export default function App() {
  const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string, sources?: string[]}[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [namespace, setNamespace] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Upload failed (${response.status}): ${err}`);
      }
      const data = await response.json();
      setNamespace(data.namespace);
      setMessages([{ role: 'assistant', content: 'Document successfully indexed! What would you like to know about it?' }]);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload document');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !namespace) return;

    const userQuery = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, namespace }),
      });
      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Query failed (${response.status}): ${err}`);
      }
      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources.map((s: any) => s.text)
      }]);
    } catch (error) {
      console.error('Query failed:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error processing your request.' }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className='flex h-screen bg-[#f9fafb] text-slate-800 font-sans selection:bg-blue-200 selection:text-blue-900'>
      
      {/* Sidebar - Apple Inspired */}
      <aside className='w-80 bg-white/80 backdrop-blur-xl border-r border-slate-200/60 p-6 flex flex-col items-center justify-between shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)] z-10'>
        <div className='w-full'>
          <div className='flex items-center justify-center gap-3 mb-10 mt-4'>
            <div className='bg-blue-600 p-2.5 rounded-2xl shadow-lg shadow-blue-600/20'>
              <BookOpen className='w-6 h-6 text-white' />
            </div>
            <h1 className='text-2xl font-bold tracking-tight text-slate-900'>Context<span className='text-blue-600'>IQ</span></h1>
          </div>

          <div className='w-full p-5 bg-slate-50 border border-slate-200 rounded-3xl mb-6 relative overflow-hidden group hover:border-blue-300 transition-colors duration-300'>
            <div className='absolute inset-0 bg-gradient-to-br from-blue-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none' />
            <h2 className='text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wider'>Knowledge Base</h2>
            <p className='text-sm text-slate-500 mb-4 leading-relaxed'>Upload a PDF document to begin an interactive, context-aware conversation.</p>
            
            <input 
              type='file' 
              accept='.pdf' 
              className='hidden' 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
            />
            
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className='w-full py-3 px-4 bg-white border shadow-sm border-slate-200 text-slate-700 rounded-xl hover:bg-slate-50 hover:text-blue-600 hover:border-blue-200 transition-all duration-200 flex items-center justify-center gap-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]'
            >
              {isUploading ? (
                <Loader2 className='w-5 h-5 animate-spin text-blue-600' />
              ) : (
                <>
                  <Upload className='w-4 h-4' />
                  Select PDF Document
                </>
              )}
            </button>
            
            {namespace && (
              <div className='mt-4 flex items-center gap-2 text-xs font-medium text-emerald-600 bg-emerald-50 py-1.5 px-3 rounded-lg border border-emerald-100'>
                <div className='w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse' />
                Document Indexed Successfully
              </div>
            )}
          </div>
        </div>

        <div className='text-xs text-slate-400 text-center flex items-center gap-1.5 bg-slate-100 py-2 px-4 rounded-full'>
          <Info className='w-3.5 h-3.5' />
          Public demo. Do not upload sensitive data.
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className='flex-1 flex flex-col h-screen relative bg-[#f9fafb]'>
        
        {/* Chat Header */}
        <header className='h-16 border-b border-slate-200/60 bg-white/60 backdrop-blur-md sticky top-0 z-10 flex items-center px-8 shadow-sm'>
          <h2 className='text-sm font-medium text-slate-600 flex items-center gap-2'>
            <div className='w-2 h-2 rounded-full bg-indigo-400' />
            Active Session
          </h2>
        </header>

        {/* Messages */}
        <div className='flex-1 overflow-y-auto px-4 py-8 scroll-smooth'>
          <div className='max-w-3xl mx-auto space-y-8'>
            
            {messages.length === 0 ? (
              <div className='flex flex-col items-center justify-center h-full mt-32 text-center'>
                 <div className='w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6 shadow-[0_0_40px_-10px_rgba(37,99,235,0.2)]'>
                    <Bot className='w-10 h-10 text-blue-500' />
                 </div>
                 <h2 className='text-2xl font-semibold text-slate-800 mb-2'>Welcome to ContextIQ</h2>
                 <p className='text-slate-500 max-w-md'>Your intelligent document assistant. Upload a PDF on the left to start analyzing context.</p>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {messages.map((m, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                    className={`flex gap-4 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {m.role === 'assistant' && (
                      <div className='w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-md flex-shrink-0 mt-1'>
                        <Bot className='w-4 h-4 text-white' />
                      </div>
                    )}
                    
                    <div className={`max-w-[85%] rounded-3xl px-6 py-4 shadow-sm ${
                      m.role === 'user'
                        ? 'bg-blue-600 text-white rounded-br-sm'
                        : 'bg-white border border-slate-200 text-slate-800 rounded-bl-sm'
                    }`}>
                      <p className='leading-relaxed whitespace-pre-wrap'>{m.content}</p>
                      
                      {m.sources && m.sources.length > 0 && (
                        <div className='mt-4 pt-4 border-t border-slate-100'>
                          <p className='text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3'>Context Sources</p>
                          <div className='space-y-2'>
                            {m.sources.map((src, sIdx) => (
                              <div key={sIdx} className='bg-slate-50 rounded-xl p-3 text-sm text-slate-600 border border-slate-100 hover:border-blue-100 hover:bg-blue-50/30 transition-colors'>
                                {src.substring(0, 150)}...
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {m.role === 'user' && (
                      <div className='w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center flex-shrink-0 mt-1 shadow-md'>
                        <User className='w-4 h-4 text-white' />
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            )}

            {isTyping && (
               <motion.div 
                 initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                 className='flex gap-4 justify-start'
               >
                 <div className='w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shadow-md flex-shrink-0 mt-1'>
                    <Bot className='w-4 h-4 text-white' />
                 </div>
                 <div className='bg-white border border-slate-200 rounded-3xl rounded-bl-sm px-5 py-4 shadow-sm flex items-center gap-1.5'>
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                    <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.4 }} className='w-2 h-2 bg-slate-300 rounded-full' />
                 </div>
               </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className='p-6 bg-transparent'>
          <div className='max-w-3xl mx-auto'>
            <form onSubmit={handleSendMessage} className='relative group'>
              <div className='absolute inset-0 bg-blue-400/5 rounded-2xl blur-xl transition-all duration-300 group-hover:bg-blue-400/10 pointer-events-none' />
              <input
                type='text'
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={!namespace || isTyping}
                placeholder={namespace ? 'Ask a question about the document...' : 'Upload a PDF to start asking...'}
                className='w-full pl-6 pr-14 py-4 bg-white border border-slate-200 text-slate-800 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] focus:outline-none focus:ring-4 focus:ring-blue-500/10 focus:border-blue-400 transition-all disabled:opacity-60 relative z-10 text-base placeholder:text-slate-400'
              />
              <button
                type='submit'
                disabled={!input.trim() || !namespace || isTyping}
                className='absolute right-2.5 top-1/2 -translate-y-1/2 p-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-all z-20 shadow-md shadow-blue-600/20 active:scale-95'
              >
                <Send className='w-4 h-4' />
              </button>
            </form>
            <div className='text-center mt-3 text-xs text-slate-400 font-medium'>
              ContextIQ AI may generate inaccurate information. Verify source references.
            </div>
          </div>
        </div>
      </main>

    </div>
  );
}
