import React, { useState } from 'react';

const PRESET_AGENTS = [
  { name: "http://localhost:10001", url: "http://localhost:10001" },
  { name: "http://localhost:10002", url: "http://localhost:10002" },
  { name: "http://localhost:10003", url: "http://localhost:10003" },
  { name: "http://localhost:10004", url: "http://localhost:10004" },
  { name: "http://localhost:10005", url: "http://localhost:10005" },
  { name: "http://localhost:10006", url: "http://localhost:10006" },
  { name: "http://localhost:10012", url: "http://localhost:10012" },
];

function AgentSelector({ onAgentSelect, isLoading }) {
  const [customUrl, setCustomUrl] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('');

  const handleSelectPreset = (e) => {
    const url = e.target.value;
    setSelectedPreset(url);
    setCustomUrl(url);
    if (url) {
      onAgentSelect(url);
    }
  };

  const handleCustomUrlChange = (e) => {
    setCustomUrl(e.target.value);
    setSelectedPreset('');
  };

  const handleSubmitUrl = (e) => {
    e.preventDefault();
    if (customUrl) {
      onAgentSelect(customUrl);
    }
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg transition-all hover:shadow-xl">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Select Agent</h2>
      <div className="mb-5">
        <label htmlFor="presetAgent" className="block text-sm font-semibold text-gray-700 mb-2">
          Preset Agents
        </label>
        <select
          id="presetAgent"
          value={selectedPreset}
          onChange={handleSelectPreset}
          className="w-full p-3 border border-gray-200 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-base disabled:bg-gray-100"
          disabled={isLoading}
        >
          <option value="">-- Select a preset --</option>
          {PRESET_AGENTS.map(agent => (
            <option key={agent.url} value={agent.url}>{agent.name}</option>
          ))}
        </select>
      </div>
      <form onSubmit={handleSubmitUrl} className="space-y-4">
        <div>
          <label htmlFor="customAgentUrl" className="block text-sm font-semibold text-gray-700 mb-2">
            Or Enter Agent URL
          </label>
          <input
            type="text"
            id="customAgentUrl"
            value={customUrl}
            onChange={handleCustomUrlChange}
            placeholder="e.g., http://localhost:10003"
            className="w-full p-3 border border-gray-200 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-base placeholder-gray-400 disabled:bg-gray-100"
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
          disabled={isLoading || !customUrl}
        >
          {isLoading ? 'Loading Card...' : 'Load Agent Card'}
        </button>
      </form>
    </div>
  );
}

export default AgentSelector;