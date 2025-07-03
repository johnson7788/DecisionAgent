import React, { useState, useCallback } from 'react';
import AgentSelector from './components/AgentSelector';
import AgentCardDisplay from './components/AgentCardDisplay';
import ChatInterface from './components/ChatInterface';
import { getAgentCard } from './services/a2aApiService';

function App() {
  const [agentUrl, setAgentUrl] = useState('');
  const [agentCard, setAgentCard] = useState(null);
  const [isLoadingCard, setIsLoadingCard] = useState(false);
  const [errorCard, setErrorCard] = useState(null);

  const handleAgentSelect = useCallback(async (url) => {
    if (!url) {
      setAgentCard(null);
      setErrorCard(null);
      setAgentUrl('');
      return;
    }
    setAgentUrl(url);
    setIsLoadingCard(true);
    setErrorCard(null);
    setAgentCard(null);
    try {
      const card = await getAgentCard(url);
      console.log(`Agent card loaded successfully: ${JSON.stringify(card)}`)
      setAgentCard(card);
    } catch (err) {
      setErrorCard(`Failed to load agent card: ${err.message}`);
      setAgentCard(null);
    } finally {
      setIsLoadingCard(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <header className="mb-10 text-center">
        <h1 className="text-5xl font-extrabold text-indigo-800 tracking-tight">A2A Single Agent Client</h1>
        <p className="mt-2 text-lg text-gray-600">Connect and interact with your agents seamlessly</p>
      </header>

      <div className="max-w-5xl mx-auto space-y-8">
        <AgentSelector onAgentSelect={handleAgentSelect} isLoading={isLoadingCard} />

        {isLoadingCard && (
          <div className="flex flex-col items-center p-6 bg-white rounded-xl shadow-lg">
            <svg className="animate-spin h-10 w-10 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="mt-4 text-lg font-medium text-indigo-600">Loading Agent Information...</p>
          </div>
        )}
        {errorCard && (
          <div className="p-6 bg-red-50 text-red-700 rounded-xl shadow-md text-base font-medium">
            {errorCard}
          </div>
        )}

        {agentCard && !errorCard && (
          <>
            <AgentCardDisplay agentCard={agentCard} />
            <ChatInterface agentCard={agentCard} />
          </>
        )}

        {!agentCard && !isLoadingCard && !errorCard && (
          <div className="p-12 text-center bg-white rounded-xl shadow-lg border-2 border-dashed border-gray-200">
            <p className="text-2xl font-semibold text-gray-500">Please select or enter an Agent URL to begin.</p>
          </div>
        )}
      </div>
      <footer className="mt-16 py-6 text-center text-gray-500 text-sm font-medium">
        A2A Client Demo &mdash; Single Agent
      </footer>
    </div>
  );
}

export default App;