'use client';

import { useState } from 'react';

interface Component {
  fraction: number;
  properties: number[];
}

interface Predictions {
  [key: string]: number;
}

export default function Home() {
  const [components, setComponents] = useState<Component[]>(
    Array(5).fill(null).map(() => ({
      fraction: 0,
      properties: Array(10).fill(0)
    }))
  );

  const [predictions, setPredictions] = useState<Predictions | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateComponent = (compIndex: number, field: 'fraction' | 'property', value: number, propIndex?: number) => {
    const newComponents = [...components];
    if (field === 'fraction') {
      newComponents[compIndex].fraction = value;
    } else if (field === 'property' && propIndex !== undefined) {
      newComponents[compIndex].properties[propIndex] = value;
    }
    setComponents(newComponents);
  };

  const generateRandomData = () => {
    // Generate random fractions that sum to 1.0
    const randomFractions: number[] = [];
    let remaining = 1.0;

    for (let i = 0; i < 4; i++) {
      const value = Math.random() * remaining;
      randomFractions.push(parseFloat(value.toFixed(3)));
      remaining -= value;
    }
    randomFractions.push(parseFloat(remaining.toFixed(3)));

    // Shuffle the fractions
    randomFractions.sort(() => Math.random() - 0.5);

    // Generate random components with properties between -10 and 10
    const newComponents = randomFractions.map(fraction => ({
      fraction,
      properties: Array(10).fill(0).map(() =>
        parseFloat((Math.random() * 20 - 10).toFixed(2))
      )
    }));

    setComponents(newComponents);
    setPredictions(null); // Clear previous predictions
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ components })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const totalFraction = components.reduce((sum, c) => sum + c.fraction, 0);

  return (
    <div className="min-h-screen p-8 font-mono">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-light mb-2 tracking-wide text-gray-100">Blend Property Predictor</h1>
          <p className="text-sm text-gray-400 font-light">Machine learning powered material blending analysis</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Input Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Fraction Summary Card with Random Generator */}
            <div className="glass rounded-3xl p-6">
              <div className="flex justify-between items-center mb-4">
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-light text-gray-300">Total Fraction</span>
                    <span className={`text-2xl font-light ${totalFraction === 1 ? 'text-green-400' : 'text-orange-400'}`}>
                      {totalFraction.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-300"
                      style={{ width: `${Math.min(totalFraction * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              <button
                onClick={generateRandomData}
                className="w-full glass rounded-2xl py-3 font-light text-sm text-gray-200
                         hover:bg-white/10 transition-all duration-200 hover:scale-[1.01]
                         border border-white/20"
              >
                🎲 Generate Random Values
              </button>
            </div>

            {/* Component Inputs */}
            {components.map((component, compIndex) => (
              <div key={compIndex} className="glass rounded-3xl p-6">
                <h3 className="text-lg font-light mb-4 text-gray-200">Component {compIndex + 1}</h3>

                {/* Fraction Input */}
                <div className="mb-4">
                  <label className="block text-xs font-light mb-2 text-gray-400">
                    Fraction
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={component.fraction}
                    onChange={(e) => updateComponent(compIndex, 'fraction', parseFloat(e.target.value) || 0)}
                    className="w-full px-4 py-3 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/20 
                             focus:outline-none focus:ring-2 focus:ring-blue-500/50 font-light text-gray-200
                             transition-all duration-200"
                  />
                </div>

                {/* Properties Grid */}
                <div className="grid grid-cols-5 gap-2">
                  {component.properties.map((prop, propIndex) => (
                    <div key={propIndex}>
                      <label className="block text-xs font-light mb-1 text-gray-400">
                        P{propIndex + 1}
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={prop}
                        onChange={(e) => updateComponent(compIndex, 'property', parseFloat(e.target.value) || 0, propIndex)}
                        className="w-full px-2 py-2 rounded-xl bg-white/5 backdrop-blur-sm border border-white/20 
                                 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 font-light text-sm text-gray-200
                                 transition-all duration-200"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={loading || totalFraction !== 1}
              className="w-full glass-strong rounded-2xl py-4 font-light text-lg text-gray-200
                       hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed
                       transition-all duration-300 hover:scale-[1.02]"
            >
              {loading ? 'Predicting...' : 'Generate Predictions'}
            </button>

            {error && (
              <div className="glass rounded-2xl p-4 bg-red-900/20 border-red-500/30">
                <p className="text-red-400 text-sm font-light">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            <div className="glass-strong rounded-3xl p-6 sticky top-8">
              <h2 className="text-2xl font-light mb-6 text-gray-200">Predictions</h2>

              {predictions ? (
                <div className="space-y-3">
                  {Object.entries(predictions).map(([property, value], index) => (
                    <div
                      key={property}
                      className="glass rounded-2xl p-4 hover:bg-white/10 transition-all duration-200"
                      style={{ animationDelay: `${index * 0.1}s` }}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-light text-gray-400">
                          {property.replace('BlendProperty', 'Property ')}
                        </span>
                        <span className="text-xl font-light tabular-nums text-gray-200">
                          {value.toFixed(4)}
                        </span>
                      </div>
                      <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-400 transition-all duration-500"
                          style={{ width: `${Math.min(Math.abs(value) * 10, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4 opacity-20">📊</div>
                  <p className="text-sm font-light text-gray-500">
                    Enter component data and click predict
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

