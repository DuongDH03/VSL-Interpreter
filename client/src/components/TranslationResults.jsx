import React from 'react';

/**
 * Component to display translation results and confidence levels
 */
export default function TranslationResults({ 
  translation, 
  confidence, 
  processingStatus, 
  error 
}) {
  return (
    <section className="bg-white rounded-xl shadow-md p-6 h-full">
      <h2 className="text-2xl font-bold text-gray-700 mb-6">Translation Results</h2>
      
      {translation ? (
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <h3 className="text-lg font-semibold text-gray-800">Detected Sign:</h3>
          <p className="text-3xl font-bold text-blue-700 my-4">{translation}</p>
          
          {confidence > 0 && (
            <div className="mt-4">
              <p className="text-sm text-gray-600">Confidence: {(confidence * 100).toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      ) : processingStatus === 'processing' ? (
        <div className="flex flex-col items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Analyzing sign language...</p>
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-600 mb-4">No sign language detected yet.</p>
          <p className="text-gray-600">Try using the webcam or uploading a video.</p>
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
