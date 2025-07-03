import React from 'react';

function AgentCardDisplay({ agentCard }) {
  if (!agentCard) {
    return null;
  }

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg transition-all hover:shadow-xl">
      <h2 className="text-2xl font-bold text-indigo-800 mb-4">{agentCard.name} &mdash; Agent Details</h2>
      <div className="space-y-4 text-gray-700 text-base">
        <p><strong className="font-semibold">Description:</strong> {agentCard.description || 'N/A'}</p>
        <p><strong className="font-semibold">Version:</strong> {agentCard.version}</p>
        <p>
          <strong className="font-semibold">Agent Endpoint URL:</strong>{' '}
          <code className="bg-gray-100 px-2 py-1 rounded-md text-sm text-gray-800">{agentCard.agentEndpointUrl}</code>
        </p>
        <p>
          <strong className="font-semibold">Documentation:</strong>{' '}
          {agentCard.documentationUrl ? (
            <a
              href={agentCard.documentationUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 hover:text-indigo-800 transition-colors"
            >
              {agentCard.documentationUrl}
            </a>
          ) : (
            'N/A'
          )}
        </p>
        <div>
          <strong className="font-semibold">Capabilities:</strong>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>Streaming: {agentCard.capabilities.streaming ? 'Yes' : 'No'}</li>
            <li>Push Notifications: {agentCard.capabilities.pushNotifications ? 'Yes' : 'No'}</li>
            <li>State Transition History: {agentCard.capabilities.stateTransitionHistory ? 'Yes' : 'No'}</li>
          </ul>
        </div>
        {agentCard.provider && (
          <p>
            <strong className="font-semibold">Provider:</strong> {agentCard.provider.organization}{' '}
            {agentCard.provider.url && (
              <a
                href={agentCard.provider.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-600 hover:text-indigo-800 transition-colors"
              >
                (Website)
              </a>
            )}
          </p>
        )}
        <p><strong className="font-semibold">Default Input Modes:</strong> {agentCard.defaultInputModes.join(', ')}</p>
        <p><strong className="font-semibold">Default Output Modes:</strong> {agentCard.defaultOutputModes.join(', ')}</p>
        {agentCard.skills && agentCard.skills.length > 0 && (
            <div>
              <strong className="font-semibold">Skills:</strong>
              <ul className="space-y-2 mt-2 ml-4">
                {agentCard.skills.map((skill, index) => (
                  <li key={index} className="border-l-4 border-indigo-500 pl-4">
                    <p><strong>Name:</strong> {skill.name}</p>
                    <p><strong>Description:</strong> {skill.description}</p>
                    <p><strong>Tags:</strong> {skill.tags?.join(', ') || 'N/A'}</p>
                    {skill.examples && skill.examples.length > 0 && (
                      <div>
                        <strong>Examples:</strong>
                        <ul className="list-disc list-inside ml-4">
                          {skill.examples.map((ex, i) => (
                            <li key={i}>{ex}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>
    </div>
  );
}

export default AgentCardDisplay;