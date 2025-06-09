import React from 'react';

/**
 * Component to display translation results and confidence levels
 */
export default function TranslationResults({ 
  translation, 
  confidence, 
  processingStatus, 
  error,
  detectedGestures = []
}) {
  return (
    <section className="bg-white rounded-xl shadow-md p-6 h-full flex flex-col">
      <h2 className="text-2xl font-bold text-gray-700 mb-4">Translation Results</h2>
      
      {translation ? (
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Detected Sign:</h3>
          <p className="text-3xl font-bold text-blue-700 my-3">{translation}</p>
          
          {confidence > 0 && (
            <div className="mt-3">
              <p className="text-sm text-gray-600">Confidence: {(confidence * 100).toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: `${Math.min(confidence * 100, 100)}%` }}
                  title={`${(confidence * 100).toFixed(1)}%`}
                ></div>
              </div>
            </div>
          )}
        </div>
      ) : processingStatus === 'processing' ? (
        <div className="flex flex-col items-center justify-center h-40 mb-4">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Analyzing sign language...</p>
        </div>
      ) : (
        <div className="text-center py-8 mb-4">
          <p className="text-gray-600 mb-4">No sign language detected yet.</p>
          <p className="text-gray-600">Try using the webcam or uploading a video.</p>
        </div>
      )}

      {/* Recent Detections History */}
      {detectedGestures.length > 0 && (
        <div className="mt-auto">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Recent Detections</h3>
          <div className="overflow-y-auto max-h-64 border border-gray-200 rounded-lg">
            <table className="min-w-full bg-white">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Gesture</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {detectedGestures.map((item, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="px-4 py-2 text-sm font-medium text-gray-800">{item.gesture}</td>
                    <td className="px-4 py-2 text-sm text-gray-600">{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="px-4 py-2 text-sm text-gray-600">{item.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Status Notifications */}
      {processingStatus === 'processing' && (
        <div className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          Processing sign language...
        </div>
      )}
      {processingStatus === 'success' && (
        <div className="fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          Translation complete!
        </div>
      )}
      {processingStatus === 'error' && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          Error: {error || 'Failed to process sign language'}
        </div>
      )}
    </section>
  );
}
