import { Level, System } from "@/types/game";

export const levels: Level[] = [
  {
    id: 1,
    name: "Station Startup",
    description: "Power up essential systems to bring the station online.",
    totalPower: 100,
    systems: [
      {
        id: "life-support",
        name: "Life Support",
        minPower: 30,
        maxPower: 50,
        currentPower: 0,
        priority: "critical",
        description: "Oxygen generation and circulation. Must be active.",
      },
      {
        id: "communications",
        name: "Communications",
        minPower: 20,
        maxPower: 40,
        currentPower: 0,
        priority: "critical",
        description: "Contact with mission control. Must be active.",
      },
      {
        id: "lighting",
        name: "Lighting",
        minPower: 10,
        maxPower: 30,
        currentPower: 0,
        priority: "important",
        description: "Station illumination. Optional but recommended.",
      },
      {
        id: "research",
        name: "Research Lab",
        minPower: 15,
        maxPower: 35,
        currentPower: 0,
        priority: "optional",
        description: "Scientific experiments. Can remain offline.",
      },
    ],
    successCondition: (systems: System[]) => {
      const lifeSupport = systems.find(s => s.id === "life-support");
      const communications = systems.find(s => s.id === "communications");

      return (
        lifeSupport !== undefined &&
        lifeSupport.currentPower >= lifeSupport.minPower &&
        communications !== undefined &&
        communications.currentPower >= communications.minPower
      );
    },
  },
  {
    id: 2,
    name: "Power Crisis",
    description: "Emergency! Distribute limited power to prevent total system failure.",
    totalPower: 80,
    systems: [
      {
        id: "life-support",
        name: "Life Support",
        minPower: 25,
        maxPower: 45,
        currentPower: 0,
        priority: "critical",
        description: "Essential for survival.",
      },
      {
        id: "shields",
        name: "Shields",
        minPower: 20,
        maxPower: 40,
        currentPower: 0,
        priority: "critical",
        description: "Protection from debris. Required.",
      },
      {
        id: "navigation",
        name: "Navigation",
        minPower: 15,
        maxPower: 30,
        currentPower: 0,
        priority: "important",
        description: "Course correction systems.",
        dependencies: ["shields"],
      },
      {
        id: "medical",
        name: "Medical Bay",
        minPower: 10,
        maxPower: 25,
        currentPower: 0,
        priority: "important",
        description: "Medical facilities for the crew.",
      },
      {
        id: "entertainment",
        name: "Entertainment",
        minPower: 5,
        maxPower: 15,
        currentPower: 0,
        priority: "optional",
        description: "Crew morale systems.",
      },
    ],
    successCondition: (systems: System[]) => {
      const lifeSupport = systems.find(s => s.id === "life-support");
      const shields = systems.find(s => s.id === "shields");

      return (
        lifeSupport !== undefined &&
        lifeSupport.currentPower >= lifeSupport.minPower &&
        shields !== undefined &&
        shields.currentPower >= shields.minPower
      );
    },
  },
  {
    id: 3,
    name: "Maximum Efficiency",
    description: "Optimize power distribution across all systems with strict constraints.",
    totalPower: 120,
    systems: [
      {
        id: "reactor-core",
        name: "Reactor Core",
        minPower: 40,
        maxPower: 60,
        currentPower: 0,
        priority: "critical",
        description: "Main power source. Must be stable.",
      },
      {
        id: "cooling",
        name: "Cooling System",
        minPower: 25,
        maxPower: 40,
        currentPower: 0,
        priority: "critical",
        description: "Prevents reactor overload.",
        dependencies: ["reactor-core"],
      },
      {
        id: "sensors",
        name: "Sensor Array",
        minPower: 15,
        maxPower: 30,
        currentPower: 0,
        priority: "critical",
        description: "Threat detection systems.",
      },
      {
        id: "weapons",
        name: "Weapons",
        minPower: 20,
        maxPower: 35,
        currentPower: 0,
        priority: "important",
        description: "Defensive systems.",
        dependencies: ["sensors"],
      },
      {
        id: "cargo",
        name: "Cargo Bay",
        minPower: 10,
        maxPower: 20,
        currentPower: 0,
        priority: "important",
        description: "Loading and storage operations.",
      },
      {
        id: "hydro",
        name: "Hydroponics",
        minPower: 8,
        maxPower: 18,
        currentPower: 0,
        priority: "optional",
        description: "Food production facility.",
      },
    ],
    successCondition: (systems: System[]) => {
      const reactor = systems.find(s => s.id === "reactor-core");
      const cooling = systems.find(s => s.id === "cooling");
      const sensors = systems.find(s => s.id === "sensors");

      return (
        reactor !== undefined &&
        reactor.currentPower >= reactor.minPower &&
        cooling !== undefined &&
        cooling.currentPower >= cooling.minPower &&
        sensors !== undefined &&
        sensors.currentPower >= sensors.minPower
      );
    },
  },
];
