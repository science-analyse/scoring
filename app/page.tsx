"use client";

import { useState, useEffect } from "react";
import { System, Level } from "@/types/game";
import { levels } from "@/lib/levels";

export default function EnergyManagementPuzzle() {
  const [currentLevelIndex, setCurrentLevelIndex] = useState(0);
  const [systems, setSystems] = useState<System[]>([]);
  const [powerUsed, setPowerUsed] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const currentLevel: Level = levels[currentLevelIndex];

  useEffect(() => {
    setSystems(JSON.parse(JSON.stringify(currentLevel.systems)));
    setIsComplete(false);
    setShowSuccess(false);
  }, [currentLevelIndex]);

  useEffect(() => {
    const totalPower = systems.reduce((sum, sys) => sum + sys.currentPower, 0);
    setPowerUsed(totalPower);

    const levelComplete = currentLevel.successCondition(systems);
    setIsComplete(levelComplete);
  }, [systems, currentLevel]);

  const handlePowerChange = (systemId: string, value: number) => {
    setSystems(prevSystems =>
      prevSystems.map(sys =>
        sys.id === systemId ? { ...sys, currentPower: value } : sys
      )
    );
  };

  const checkSolution = () => {
    if (powerUsed <= currentLevel.totalPower && isComplete) {
      setShowSuccess(true);
    } else {
      alert("Solution doesn't meet requirements. Check critical systems!");
    }
  };

  const nextLevel = () => {
    if (currentLevelIndex < levels.length - 1) {
      setCurrentLevelIndex(prev => prev + 1);
    }
  };

  const resetLevel = () => {
    setSystems(JSON.parse(JSON.stringify(currentLevel.systems)));
  };

  const isSystemActive = (system: System): boolean => {
    return system.currentPower >= system.minPower;
  };

  const areDependenciesMet = (system: System): boolean => {
    if (!system.dependencies || system.dependencies.length === 0) return true;

    return system.dependencies.every(depId => {
      const depSystem = systems.find(s => s.id === depId);
      return depSystem && isSystemActive(depSystem);
    });
  };

  const getPriorityColor = (priority: string): string => {
    switch (priority) {
      case "critical":
        return "text-red-400";
      case "important":
        return "text-yellow-400";
      case "optional":
        return "text-blue-400";
      default:
        return "text-gray-400";
    }
  };

  const getSystemStatusColor = (system: System): string => {
    if (!areDependenciesMet(system)) return "bg-gray-700";
    if (isSystemActive(system)) return "bg-green-600";
    return "bg-red-900";
  };

  const powerOverage = powerUsed - currentLevel.totalPower;
  const isPowerExceeded = powerOverage > 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Energy Management Puzzle
          </h1>
          <p className="text-gray-400 text-lg">Allocate limited power across systems</p>
        </div>

        {/* Level Info */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-2xl font-bold text-blue-400">
                Level {currentLevel.id}: {currentLevel.name}
              </h2>
              <p className="text-gray-300 mt-2">{currentLevel.description}</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400">Progress</div>
              <div className="text-xl font-bold">
                {currentLevelIndex + 1} / {levels.length}
              </div>
            </div>
          </div>

          {/* Power Display */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">Power Available</div>
              <div className="text-3xl font-bold text-cyan-400">
                {currentLevel.totalPower} MW
              </div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">Power Used</div>
              <div
                className={`text-3xl font-bold ${
                  isPowerExceeded ? "text-red-500" : "text-green-400"
                }`}
              >
                {powerUsed} MW
              </div>
            </div>
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">Remaining</div>
              <div
                className={`text-3xl font-bold ${
                  isPowerExceeded ? "text-red-500" : "text-yellow-400"
                }`}
              >
                {isPowerExceeded ? `+${powerOverage}` : currentLevel.totalPower - powerUsed} MW
              </div>
            </div>
          </div>

          {/* Power Bar */}
          <div className="mt-4">
            <div className="bg-gray-700 rounded-full h-4 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  isPowerExceeded
                    ? "bg-red-500"
                    : powerUsed === currentLevel.totalPower
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
                style={{
                  width: `${Math.min((powerUsed / currentLevel.totalPower) * 100, 100)}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* Systems Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {systems.map(system => {
            const active = isSystemActive(system);
            const depsMet = areDependenciesMet(system);

            return (
              <div
                key={system.id}
                className={`bg-gray-800 rounded-lg p-5 border-2 transition-all ${
                  active && depsMet
                    ? "border-green-500"
                    : !depsMet
                    ? "border-gray-600"
                    : "border-red-500"
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="text-xl font-bold">{system.name}</h3>
                    <p className={`text-sm ${getPriorityColor(system.priority)}`}>
                      {system.priority.toUpperCase()}
                    </p>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-bold ${getSystemStatusColor(
                      system
                    )}`}
                  >
                    {!depsMet
                      ? "LOCKED"
                      : active
                      ? "ACTIVE"
                      : "OFFLINE"}
                  </div>
                </div>

                <p className="text-gray-400 text-sm mb-4">{system.description}</p>

                {system.dependencies && system.dependencies.length > 0 && (
                  <div className="mb-3 text-sm text-orange-400">
                    Requires: {system.dependencies.map(depId => {
                      const depSystem = systems.find(s => s.id === depId);
                      return depSystem?.name;
                    }).join(", ")}
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">
                      Range: {system.minPower}-{system.maxPower} MW
                    </span>
                    <span className="font-bold text-cyan-400">
                      {system.currentPower} MW
                    </span>
                  </div>

                  <input
                    type="range"
                    min="0"
                    max={system.maxPower}
                    value={system.currentPower}
                    onChange={e =>
                      handlePowerChange(system.id, parseInt(e.target.value))
                    }
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />

                  <div className="flex gap-2">
                    <button
                      onClick={() => handlePowerChange(system.id, 0)}
                      className="flex-1 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                    >
                      Off
                    </button>
                    <button
                      onClick={() =>
                        handlePowerChange(system.id, system.minPower)
                      }
                      className="flex-1 px-3 py-1 bg-blue-700 hover:bg-blue-600 rounded text-sm transition-colors"
                    >
                      Min
                    </button>
                    <button
                      onClick={() =>
                        handlePowerChange(system.id, system.maxPower)
                      }
                      className="flex-1 px-3 py-1 bg-purple-700 hover:bg-purple-600 rounded text-sm transition-colors"
                    >
                      Max
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          <button
            onClick={resetLevel}
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-bold transition-colors"
          >
            Reset Level
          </button>
          <button
            onClick={checkSolution}
            disabled={isPowerExceeded}
            className={`px-8 py-3 rounded-lg font-bold transition-all ${
              isPowerExceeded
                ? "bg-gray-600 cursor-not-allowed"
                : isComplete
                ? "bg-green-600 hover:bg-green-500 animate-pulse"
                : "bg-blue-600 hover:bg-blue-500"
            }`}
          >
            {isPowerExceeded
              ? "Power Exceeded!"
              : isComplete
              ? "Submit Solution ✓"
              : "Check Solution"}
          </button>
        </div>

        {/* Success Modal */}
        {showSuccess && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-8 max-w-md border-2 border-green-500">
              <h2 className="text-3xl font-bold text-green-400 mb-4">
                Level Complete!
              </h2>
              <p className="text-gray-300 mb-6">
                Excellent power management! All critical systems are online.
              </p>
              <div className="flex gap-4">
                {currentLevelIndex < levels.length - 1 ? (
                  <button
                    onClick={nextLevel}
                    className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-500 rounded-lg font-bold transition-colors"
                  >
                    Next Level →
                  </button>
                ) : (
                  <div className="text-center w-full">
                    <p className="text-2xl font-bold text-yellow-400 mb-4">
                      Congratulations! All levels completed!
                    </p>
                    <button
                      onClick={() => setCurrentLevelIndex(0)}
                      className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-bold transition-colors"
                    >
                      Play Again
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
