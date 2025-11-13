'use client';

import { useMemo, useEffect, useState } from 'react';

export interface AlgorithmOption {
  name: string;
  description: string;
  available?: boolean;
}

interface AlgorithmSelectorProps {
  algorithms: AlgorithmOption[];
  selected?: string;
  description?: string;
  onSelect: (name: string) => void;
}

const FALLBACK_ALGORITHMS: AlgorithmOption[] = [
  { name: 'First Come First Serve (FCFS)', description: '' },
  { name: 'Shortest Job First (SJF)', description: '' },
  { name: 'Priority Scheduling', description: '' },
  { name: 'Multilevel Feedback Queue (MLFQ)', description: '' },
  { name: 'Compare All', description: 'View all algorithms side-by-side and see which performs best overall based on average waiting time.' }
];

export function AlgorithmSelector({
  algorithms,
  selected,
  description,
  onSelect
}: AlgorithmSelectorProps) {
  const availableOptions = algorithms.length ? algorithms : FALLBACK_ALGORITHMS;
  const [animatedChars, setAnimatedChars] = useState<boolean[]>([]);

  useEffect(() => {
    const text = 'SCHEDULING ALGORITHM';
    
    const animate = () => {
      setAnimatedChars(new Array(text.length).fill(false));
      
      // Animate each character with a delay
      text.split('').forEach((_, i) => {
        setTimeout(() => {
          setAnimatedChars((prev) => {
            const newState = [...prev];
            newState[i] = true;
            return newState;
          });
        }, i * 50);
      });
    };

    // Initial animation
    animate();

    // Repeat animation every 12 seconds
    const interval = setInterval(animate, 12000);

    return () => clearInterval(interval);
  }, []);

  const currentDescription = useMemo(() => {
    const match = availableOptions.find((item) => item.name === selected);
    return description ?? match?.description ?? '';
  }, [availableOptions, description, selected]);

  return (
    <div className="panel" style={{ maxWidth: 420 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <p className="badge">Scheduler</p>
          <div style={{ marginTop: 12 }}>
            <div style={{ overflow: 'hidden', fontSize: 26, display: 'flex', cursor: 'default' }}>
              {'SCHEDULING ALGORITHM'.split('').map((char, i) => (
                <h2 
                  key={i} 
                  style={{ 
                    fontFamily: 'monospace', 
                    opacity: animatedChars[i] ? 1 : 0,
                    transform: animatedChars[i] ? 'translateY(0)' : 'translateY(20px)',
                    transition: 'opacity 0.3s ease, transform 0.3s ease',
                    margin: 0, 
                    width: char === ' ' ? 8 : 'auto', 
                    fontSize: 26, 
                    fontWeight: 600 
                  }}
                >
                  {char}
                </h2>
              ))}
            </div>
          </div>
        </div>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 44,
            height: 44,
            borderRadius: '50%',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}
        >
          ⚙️
        </span>
      </div>

      <div className="selector-grid">
        {availableOptions.map((option) => {
          const isSelected = option.name === selected;
          const isAvailable = option.available ?? true;
          return (
            <button
              key={option.name}
              type="button"
              className={`selector-option${isSelected ? ' active' : ''}${
                !isAvailable ? ' muted' : ''
              }`}
              onClick={() => onSelect(option.name)}
            >
              <span className="selector-title">{option.name}</span>
              <span className="selector-caption">
                {option.name === 'Compare All' ? 'View all algorithms' : (isAvailable ? 'Live data ready' : 'No active data')}
              </span>
            </button>
          );
        })}
      </div>

      {currentDescription && <p className="panel-subtitle">{currentDescription}</p>}
    </div>
  );
}
