export interface System {
  id: string;
  name: string;
  minPower: number;
  maxPower: number;
  currentPower: number;
  priority: 'critical' | 'important' | 'optional';
  description: string;
  dependencies?: string[]; // IDs of systems that must be active
}

export interface Level {
  id: number;
  name: string;
  description: string;
  totalPower: number;
  systems: System[];
  successCondition: (systems: System[]) => boolean;
}

export interface GameState {
  currentLevel: number;
  levels: Level[];
  isLevelComplete: boolean;
  powerUsed: number;
  powerAvailable: number;
}
