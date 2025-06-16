import React from 'react';

/**
 * Component for the learning mode UI
 */
export default function LearnComponent({ currentSign, isCorrect, processingStatus, error }) {
  return (
    <section className="bg-white rounded-xl shadow-md p-6 h-full flex flex-col">
      <h2 className="text-2xl font-bold text-gray-700 mb-4">Learn Sign Language</h2>
      
      {/* Current Sign to Practice */}
      <div className="bg-blue-50 p-6 rounded-lg border border-blue-200 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Current Sign:</h3>
        <p className="text-3xl font-bold text-blue-700">{currentSign || "16.MatTroi"}</p>
      </div>
      
      {/* Feedback Area */}
      {processingStatus === 'processing' ? (
        <div className="flex flex-col items-center justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Analyzing your sign...</p>
        </div>
      ) : isCorrect === true ? (
        <div className="bg-green-100 p-6 rounded-lg border border-green-300 text-center">
          <div className="text-5xl text-green-600 mb-4">✓</div>
          <p className="text-xl font-bold text-green-800">Correct!</p>
          <p className="text-gray-600 mt-2">Great job! Try again or try another sign.</p>
        </div>
      ) : isCorrect === false ? (
        <div className="bg-red-100 p-6 rounded-lg border border-red-300 text-center">
          <div className="text-5xl text-red-600 mb-4">✗</div>
          <p className="text-xl font-bold text-red-800">Not quite!</p>
          <p className="text-gray-600 mt-2">Try again!</p>
        </div>
      ) : (
        <div className="bg-gray-100 p-6 rounded-lg border border-gray-200 text-center">
          <p className="text-xl text-gray-600">Show the "{currentSign || "16.MatTroi"}" sign to the camera</p>
        </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
        </div>
      )}
      
      {/* Help Text */}
      <div className="mt-auto pt-6">
        <p className="text-sm text-gray-500">
          Practice makes perfect! The system will tell you if your sign is correct.
        </p>
      </div>
    </section>
  );
}
